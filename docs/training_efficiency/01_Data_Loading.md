## Summary
Techniques added continuously: covers the dataloader-related chain from storage → decode → CPU preprocessing → worker scheduling and memory management.
1. Preallocate per-frame output memory to fold decode into a single copy
2. Multi-camera lazy loading (lazy initialization)
3. Camera decode thread concurrency (`camera_decode_max_concurrency`)

---

## 1. Preallocate per-frame output memory to fold decode into a single copy

**Idea**:

![Before: each decoded frame is first stored in a list (4 orange blocks), then torch.stack copies them all at once into the output tensor (4 blue blocks) — peak memory is two full copies. After: decode writes each frame directly with .copy_() into its slot in the output buffer that was allocated at the first frame, so at any moment there's only "one output block plus one frame in flight," with no second full copy.](assets/01_preallocate_the_decode_output.png)

As soon as the first frame is decoded, allocate the final `[N, H, W, 3]` contiguous buffer right away; every subsequent decoded frame is `.copy_()`'d directly into its slot in that buffer, eliminating the "scatter first, stack at the end" two-step.

**Gain**: (internal benchmark, for reference) measured standalone: loader throughput **+33.9%**, stall burden **−82.7%**, per-sample load time −20~30%.

**Implementation**:

Before:
```python
collected: dict[int, torch.Tensor] = {}
# in the decode loop, each decoded frame is stored in the dict
collected[cur_frame_idx] = torch.as_tensor(frame.to_ndarray(format="rgb24"))
...
# after decoding finishes, pull them out in order and stack into one big tensor
collected = [collected[i] for i in unique_frame_idxs]
tensors = torch.stack(collected)
```

After:
```python
output_slots = {frame_idx: i for i, frame_idx in enumerate(unique_frame_idxs)}
tensors: torch.Tensor | None = None

# in the decode loop
frame_tensor = torch.as_tensor(frame.to_ndarray(format="rgb24"))
if tensors is None:
    # the frame's shape is only known once the first frame is decoded; allocate the
    # final-size buffer once, right here
    tensors = frame_tensor.new_empty((len(unique_frame_idxs), *frame_tensor.shape))
tensors[output_slots[cur_frame_idx]].copy_(frame_tensor)  # copy in-place straight into its final slot
```

If an incoming frame's size doesn't match the preallocated buffer, `copy_()` crashes outright, and `torch.stack` is monkeypatched to raise if it's ever called.

**Use when**: not limited to video decoding — apply this whenever you see a loop that keeps **pushing tensors into a list/dict and then calls `stack`/`cat` once at the end**: know the final size ahead of time → allocate the buffer once → write directly into the right slot inside the loop, eliminating that final redundant full copy.

---

## 2. Multi-camera lazy loading (lazy initialization)

**Idea**:

![Before: all 6 cameras are opened, indexed and decoded during clip admission, and cam5/cam6 (orange) are the unselected ones doing wasted work. After: all 6 cameras are only opened/indexed at first real decode — cam1-4 (blue, selected) do this work at first decode, and cam5/cam6 (gray, unselected) are never opened or decoded. Single-node CPU test: per-sample load time 1.640→1.475s (−10.1%); 90 of 204 container reads eliminated (−44%).](assets/01_open_only_the_selected_cameras.png)

This optimization is useful for any dataset that loads a varying subset of the configured cameras/sensors per sample rather than all of them — for example, a strategy like `camera_subsample_weights`, which draws each sample a weighted "camera subset plan" ahead of time (e.g. all 6 / only the first 3) as a form of data augmentation. As an example, when a sample loads an average of only about 4.1 of 6 configured cameras, the clip-admission stage would previously **eagerly open all 6 cameras' MP4 files** and build their keyframe indexes, regardless of whether a given camera would end up being used. After the change, a `DeferredVideoReader` is introduced that only records the camera's path (the `video_path` string); actually opening the file and constructing the concrete reader (`SeekVideoReader`, etc.) is **deferred until the first time that camera is actually decoded** — an unselected camera is never opened.

**Gain**: (internal benchmark, for reference) `mean_load_s_per_sample` 1.640→**1.475** (−10.1%); full-SFT A/B: whole-step wall **−5.7%**, severe stalls **−30%**; 90 of 204 camera payload reads eliminated (**−44%**).

**Implementation**:

Before:
```python
# during clip admission, eagerly open all 6 cameras
owned_handles = []
for camera_name in self._camera_names_in_order():
    reader, handle = self._build_camera_reader(camera_name)  # opens(video_path, "rb") immediately
    clip[camera_name] = reader
    owned_handles.append(handle)
```

After:
```python
class DeferredVideoReader(VideoReader):
    """Path-backed reader that opens its concrete reader on first decode."""

    def __init__(self, video_path, timestamps, reader_cls, thread_count):
        super().__init__(io.BytesIO(), timestamps, thread_count)
        self._video_path = video_path      # records only the path, doesn't open the file
        self._reader_cls = reader_cls
        self._reader: VideoReader | None = None

    @property
    def initialized(self) -> bool:
        return self._reader is not None

    def _get_reader(self) -> VideoReader:
        """Lazy-load core: only opens the file and constructs the reader on the first call, then reuses the cached one."""
        reader = self._reader
        if reader is None:                     # not yet initialized
            video_handle = open(self._video_path, "rb")           # the file is actually opened here
            reader = self._reader_cls(video_handle, self.timestamps, thread_count=self._thread_count)
            self._reader = reader               # cache it so it's not reopened next time
        return reader                           # whether just-initialized or already-initialized, returns here

    # neither public decode entry point touches the file directly; both go through
    # _get_reader() to trigger or reuse the lazy load
    def decode_images_from_timestamps(self, requested_timestamps):
        return self._get_reader().decode_images_from_timestamps(requested_timestamps)

    def decode_images_from_frame_indices(self, frame_indices):
        return self._get_reader().decode_images_from_frame_indices(frame_indices)
```

The admission stage now just puts 6 `DeferredVideoReader`s (holding no file handles) into the clip dict; nobody has actually opened a file yet. The real `SeekVideoReader` is only lazily constructed and its MP4 opened the first time `decode_images_from_frame_indices` is called.

**Use when**: anytime the cost of "initializing/preparing a resource" scales with **the full set declared in config** (e.g. 6 cameras configured here), rather than with **the subset actually used** (a sample uses 4.1 on average) — consider deferring initialization to the moment you're actually sure you need the resource. In other words, lazy loading only pays off when "declared scale > actually-used scale"; if the two are always equal (every resource really does get used every time), lazy loading has no benefit.

---

## 3. Camera decode thread concurrency (`camera_decode_max_concurrency`)

**Idea**:

![Before (cap=1): the worker thread decodes cam1-6 sequentially, and the six camera-decodes' wall time string together into one line. After (cap=2): the worker thread decodes cam1/3/5 while a reused helper thread concurrently decodes cam2/4/6 — total CPU work is unchanged, but wall time is halved, so the slowest sample's decode time is cut in half. S2 6-worker scenario: severe steps 18.27%→10.00%, mean 1.784→1.530s/step; S1 same-node pairing: severe burden −33%~−45%, clean-path cost +1.6%.](assets/01_bounded_decode_overlap_for_latency_tails.png)

The worker decoder is strictly single-threaded and serial (`video_decode_thread_count=1`, because 80 worker processes per node can't each run their own FFmpeg thread pool). A 6-camera, 4-frame sample actually needs to decode **30-50 frames** of 1080p HEVC (video is GOP-compressed, so reaching a target frame means decoding sequentially through the frames ahead of it in its GOP — more than "6×4" worth). A **"heavy" sample** — one with a noticeably higher frame count — can drag down the whole batch once it gets stuck: a worker process delivers samples in FIFO order, so when a heavy sample comes up, it must finish decoding before the next one can be handed off; and since distributed training requires every rank to have a complete batch before the step can start, if even one of the 80 workers is stuck on a heavy sample, every other rank (even ones that were already ready) has to sit and wait — this is exactly the trigger scenario for a severe stall.

The change gives each worker process a **reused helper thread** (`os.register_at_fork` guarantees fork-safety; it's process-private, built once, and reused repeatedly rather than being torn down and recreated each time): the cameras to decode are split into pairs (`camera_decode_max_concurrency=2`); within each pair, the main thread decodes the first camera while the helper thread concurrently decodes the second, with results landing strictly in input order; the next pair then proceeds the same way — it's not that all the selected cameras get thrown at 2 threads at once. The concurrency cap is locked to 2 by the schema (`Literal[1, 2]`); raising the cap to 4 was tested and found to be a regression (+12%, the SMT hardware-thread budget gets overrun).

**Gain**: (internal benchmark, for reference) tested across three scenario groups
- **S1, same-node paired A/B**: stall gap (extra time per step from stalling) **−0.050s/step**; severe burden **−33%~−45%**; side-effect check — even on the "clean path" that never hits a heavy sample, the cost is only **+1.6%** (within the ±2% node-noise range, essentially no added overhead)
- **S2, 6-worker (small worker pool, a single heavy sample more easily drags down the whole run)**: stacked with "lazy loading + bounded decode concurrency," severe steps **18.27%→10.00%** (the combined effect of "prefetch4 + lazy init + decode concurrency" together)
- **S2, 10-worker (large enough worker pool + the bottleneck is elsewhere)**: severe steps 16.5%→16.5%, no meaningful gain. Reason: ① with a large enough pool, when one worker gets stuck the others can pick up the slack, so it's no longer the bottleneck; ② the root cause of severe stalls here in S2 is actually the allocator/32MB mmap-threshold issue (see the CPU-preprocessing techniques below), a different mechanism from "heavy-sample long-tail latency."

**Implementation**:

Before:
```python
# cameras decoded strictly serially
image_data = {}
for camera_name in cameras_to_load:
    video_cache: VideoReader = clip.cached_data[camera_name]
    images, timestamps = video_cache.decode_images_from_timestamps(t0s)
    image_data[camera_name] = ImageSample(images=images, timestamps=timestamps)
```

After:
```python
class _CameraDecodeHelper:
    """Each DataLoader worker process lazily builds one reused single-thread helper executor."""
    def get(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, ...)
            return self._executor

_CAMERA_DECODE_HELPER = _CameraDecodeHelper()
os.register_at_fork(after_in_child=_CAMERA_DECODE_HELPER.reset_after_fork)  # fork-safe

def _decode_camera_pair(first_reader, second_reader, t0s):
    """The second camera is handed to the helper thread; the first camera stays on the calling thread; results return in input order."""
    second_future = _CAMERA_DECODE_HELPER.get().submit(_decode_camera, second_reader, t0s)
    first_sample = _decode_camera(first_reader, t0s)
    return first_sample, second_future.result()

# process in groups of camera_decode_max_concurrency (1 or 2)
for start in range(0, len(camera_order), group_size):
    camera_names = camera_order[start : start + group_size]
    readers = [clip.cached_data[name] for name in camera_names]
    samples = _decode_camera_pair(readers[0], readers[1], t0s) if len(readers) == 2 \
        else (_decode_camera(readers[0], t0s),)
```

`camera_decode_max_concurrency` is an explicit switch validated by pydantic config: it must be 1 or 2, and `=2` forces `video_cache_type="seek"` and `video_decode_thread_count=1` — if the config doesn't satisfy that, it errors out directly rather than silently taking effect as incorrect behavior.
