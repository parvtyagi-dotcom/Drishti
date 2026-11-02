package in.borderwatch.siren;

import android.os.Bundle;
import android.view.WindowManager;

import com.getcapacitor.BridgeActivity;

/**
 * Drishti Post Siren.
 *
 * This phone is an appliance, not a browsing session: it sits at a post and
 * its only job is to be listening when an alert arrives. Two things follow
 * from that, and both are set here rather than left to the web layer.
 */
public class MainActivity extends BridgeActivity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Keep the screen on while the app is in front. A sleeping device
        // suspends the WebView, which stops the websocket and silently turns
        // the siren into a phone that is no longer listening - the worst
        // possible failure for an alarm, because nothing appears wrong.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Show over the lock screen and wake the display when an alert fires,
        // so the alarm is visible without someone unlocking the phone first.
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
        } else {
            getWindow().addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED
                    | WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);
        }
    }
}
