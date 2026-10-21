"""Fix GLM-5.3-Flash video placeholder timestamps (upstream vLLM PR #55647).

vLLM 0.30.0's ``Glm5NextProcessingInfo`` inherits the GLM-4.6V timestamp
sampler, while the checkpoint's ``Glm5NextVideoProcessor`` samples pixels with
its own ``fps_interval`` policy. The two samplers only agree for 30-300s
clips; outside that window the placeholder count diverges from the encoder's
``video_grid_thw`` row count, and the mismatch in
``_merge_multimodal_embeddings`` kills the engine core.

This patch overrides ``_get_video_second_idx_glm46v`` to sample with the
pixel path's own policy, making the placeholder count structural: one
timestamp per temporal patch, exactly ``grid_t`` for any clip. Clips too
short to yield a single frame raise a clear ``ValueError`` instead of an
opaque ``IndexError`` deeper in the pixel path.
"""

from typing import Any

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _get_video_second_idx_glm46v(
    self, metadata: dict[str, Any], total_frames: int
) -> list[int]:
    """Timestamps for the video placeholder, from the pixel path's sampler.

    The inherited GLM-4.6V timestamp sampling uses different constants
    than this checkpoint's ``Glm5NextVideoProcessor`` pixel sampling, so
    placeholder count and ``video_grid_thw`` disagree outside 30-300s
    durations. Sampling with the pixel path's own policy makes one
    timestamp per temporal patch == ``grid_t``.
    """
    from vllm.transformers_utils.processors.glm5next import (
        glm_sample_frame_indices,
    )

    video_processor = self.get_video_processor()
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 1) or 1)
    video_fps = float(metadata["fps"])

    if not metadata.get("do_sample_frames", True):
        # The loader already picked the frames and the processor keeps them
        # as they are, so the frame count is what was handed in and
        # ``frames_indices`` only supplies the timestamp values.
        frame_indices = [int(idx) for idx in metadata["frames_indices"]]
        num_frames = int(total_frames)
    else:
        frame_indices = [
            int(idx)
            for idx in glm_sample_frame_indices(
                int(metadata.get("total_num_frames", total_frames)),
                video_fps,
                float(metadata.get("duration") or 0),
                target_fps=video_processor.fps_interval,
                max_frame_count=video_processor.max_frame_count_dynamic,
                temporal_patch_size=temporal_patch_size,
            )
        ]
        num_frames = len(frame_indices)

    if not frame_indices or num_frames <= 0:
        # ``duration * fps_interval < 1`` selects no frames at all; the
        # pixel path raises an opaque IndexError on the same input a moment
        # later, so say what happened while there is still a request.
        raise ValueError(
            "The GLM-5-Next frame sampler selected no frames for this "
            f"video (fps={video_fps}, duration={metadata.get('duration')!r},"
            f" total_num_frames={metadata.get('total_num_frames')!r}). "
            "Clips shorter than one frame sampling interval cannot be "
            "processed."
        )

    num_padded_frames = num_frames + (-num_frames % temporal_patch_size)
    grid_t = max(num_padded_frames // temporal_patch_size, 1)
    last = len(frame_indices) - 1
    return [
        int(frame_indices[min(i * temporal_patch_size, last)] / video_fps)
        for i in range(grid_t)
    ]


@when_imported("vllm.models.glm5next.nvidia.multimodal")
def patch_glm5next_video_timestamps(mod):
    """Override GLM-5.3-Flash placeholder timestamps with the pixel sampler."""
    info_cls = getattr(mod, "Glm5NextProcessingInfo", None)
    if info_cls is None:
        logger.warning(
            "Skip GLM-5.3-Flash video timestamp patch: "
            "Glm5NextProcessingInfo not found"
        )
        return

    info_cls._get_video_second_idx_glm46v = _get_video_second_idx_glm46v
    logger.info(
        "UCM GLM-5.3-Flash video placeholder timestamp patch applied: "
        "_get_video_second_idx_glm46v now samples with the pixel path's "
        "own frame policy"
    )
