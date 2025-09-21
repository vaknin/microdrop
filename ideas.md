# STT Performance Ideas

- Stream audio chunks through a worker queue to transcribe while still recording, emitting partial text instead of one big post-process.
- [done] Hand the in-memory `float32` array from the recorder directly to `self.model.transcribe` to skip temporary WAV writes and reads.
- Enable VAD and adjust decoding params (`vad_filter=True`, `beam_size=1`, `best_of=1`) to reduce time spent on silence and shrink the search space.
- Split long recordings into fixed or energy-based chunks and transcribe them in parallel threads, then merge using timestamps to keep the model cache warm without a monolithic pass.
