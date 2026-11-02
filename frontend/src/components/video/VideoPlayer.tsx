import { forwardRef, useImperativeHandle, useRef } from "react";

export interface VideoPlayerHandle {
  seekTo: (seconds: number) => void;
}

/** Thin wrapper around the real HTML5 <video> element - native controls
 * (play/pause/timeline/volume/fullscreen/seeking), no custom canvas
 * rendering. The backend output video already contains every overlay
 * (target box, zone outline, event banners) burned in, so this never draws
 * anything on top of the frame itself - only exposes a seek handle so the
 * event timeline can jump playback to a clicked event's timestamp.
 *
 * onTimeUpdate reports the playhead so a caller can react to playback
 * reaching a moment in the footage - which is how the alert siren stays tied
 * to what is on screen rather than to when the analysis finished. */
export const VideoPlayer = forwardRef<VideoPlayerHandle, {
  src: string;
  poster?: string;
  onTimeUpdate?: (seconds: number) => void;
}>(function VideoPlayer(
  { src, poster, onTimeUpdate },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useImperativeHandle(ref, () => ({
    seekTo(seconds: number) {
      const el = videoRef.current;
      if (!el) return;
      el.currentTime = seconds;
      if (el.paused) void el.play().catch(() => {});
    },
  }));

  return (
    <div className="overflow-hidden rounded-[10px] border border-border-1 bg-black">
      <video
        ref={videoRef}
        src={src}
        controls
        poster={poster}
        preload="metadata"
        onTimeUpdate={(e) => onTimeUpdate?.(e.currentTarget.currentTime)}
        onSeeked={(e) => onTimeUpdate?.(e.currentTarget.currentTime)}
        className="block max-h-[640px] w-full bg-black"
      />
    </div>
  );
});
