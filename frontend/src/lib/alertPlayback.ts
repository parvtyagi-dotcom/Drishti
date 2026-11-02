/**
 * Deciding which alerts playback has just reached.
 *
 * An alert that fires the moment analysis finishes tells an operator only
 * that the job is done. Tying the siren to the playhead makes it sound while
 * the target is actually on screen and boxed, which is the thing they are
 * being asked to look at.
 *
 * The awkward part is seeking. Dragging the scrubber across three alerts is
 * not playback arriving at three moments, and firing three sirens for it
 * would be noise; nor should dragging back and forth re-fire the same alert
 * endlessly. So a jump marks everything behind the new position as already
 * heard without sounding it, and re-arms whatever lies ahead - which also
 * makes replaying a stretch of footage sound its alerts again, as it should.
 *
 * Kept as a pure function, separate from the page, so this behaviour can be
 * checked directly instead of by scrubbing a video by hand.
 */

export interface PlaybackAlert {
  /** Stable identity for one alert moment. */
  key: string;
  timestampSec: number;
}

export interface AlertsReached {
  /** Alerts playback has just arrived at. Empty while seeking. */
  due: PlaybackAlert[];
  /** Replacement set of alerts already heard. */
  sounded: Set<string>;
}

/**
 * timeupdate fires roughly four times a second, so ordinary playback advances
 * in steps well under this. A larger forward jump means someone dragged the
 * scrubber, not that the footage ran on.
 */
export const MAX_PLAYBACK_STEP_SEC = 1.0;

export function alertsReachedBy(
  alerts: PlaybackAlert[],
  previousSec: number,
  nowSec: number,
  sounded: ReadonlySet<string>,
): AlertsReached {
  const seeked = nowSec < previousSec || nowSec - previousSec > MAX_PLAYBACK_STEP_SEC;

  if (seeked) {
    return {
      due: [],
      sounded: new Set(alerts.filter((a) => a.timestampSec <= nowSec).map((a) => a.key)),
    };
  }

  const due = alerts.filter((a) => !sounded.has(a.key) && a.timestampSec <= nowSec);
  if (due.length === 0) return { due, sounded: new Set(sounded) };

  const next = new Set(sounded);
  due.forEach((a) => next.add(a.key));
  return { due, sounded: next };
}
