# Geometric conventions

> English translation of the team's [`docs/frames.md`](../frames.md).

- **Camera frame (OpenCV)**: x to the right, y down, z forward (optical axis).
- **World frame**: origin at the optical centre of the first pose. After floor alignment (phase 6), **z up**, the plane z = 0 is the floor.
  Before alignment, the world frame coincides with the first camera.
- **Poses**: 4×4 matrices `T_world_cam` (float64); a camera point `p_c` becomes `p_w = T_world_cam @ [p_c, 1]`.
- **Units**: metres, seconds, radians (degrees only in the config, converted on reading).
- **Grid (costmap)**: indices `(i, j)` = (row, column); `x_world = origin_x + (j + 0.5) * res`, `y_world = origin_y + (i + 0.5) * res`.
- **Azimuth** (guidance, audio): angle in the horizontal plane, relative to the head, **positive to the left** (counter-clockwise seen from above, z up), 0 = straight ahead. Elevation positive upward.
- **Image**: after rotation according to the video metadata and resizing to `video.width`. The intrinsics K correspond to the rotated and resized image.
- **Time**: each message carries `t` (s, source clock = video timestamp) and `frame_idx`.

## Phase 2: phone worn on the chest

- **Device frame (W3C DeviceMotion)**: x to the right of the screen, y to the top of the screen, z out of the screen (towards the user). `rotationRate` is given in deg/s by the specification (`beta` around x, `gamma` around y, `alpha` around z): the page sends `[beta, gamma, alpha]` and the server converts to rad/s. `accelerationIncludingGravity` is in m/s²; at rest, it equals **+g along the upward axis** (phone upright in portrait: ≈ (0, +9.81, 0)).
  Warning: some browsers have inverted signs or units in the past; the phase 17 calibration checks `R_cam_imu` and the units on real data.
- **Rear camera in portrait**: `R_cam_imu = diag(1, −1, −1)` a priori (shared x, camera y down = −device y, camera z forward = −device z). `ω_cam = R_cam_imu · ω_imu`.
- **Body frame** (= head, hackathon assumption): expressed in the **camera convention** (x right, y down, z forward) to stay compatible with the existing guidance code. `T_cam_body` (config `body.T_cam_body`): identity when the phone is upright on the chest. `T_world_body = T_world_cam @ T_cam_body`. `guidance.T_cam_head` is kept and only used if `body.T_cam_body` is missing.
- **Time**:
  - `t_phone`: phone clock, `(performance.timeOrigin + performance.now()) / 1000`, the same for images and sensor events;
  - `t_recv`: computer clock (`time.time()`) on reception;
  - `ClockSync.offset`: `t_pc ≈ t_phone + offset`, estimated by ping/pong (minimal half-RTT over a sliding window).
  - The `t` of messages coming from the phone is `t_phone` (source clock, like the video timestamp in phase 1).
- **Camera height**: `body.camera_height_m`, declared by the user (measured from the floor), is used as the scale reference (phase 18).

> Note from the fork: real calibration later showed that Safari iOS does not follow the specification for gyroscope axes and accelerometer sign; see [`DECISIONS.md`](DECISIONS.md) ("Safari iOS: gyroscope axes and accelerometer sign"). On Android Chrome the accelerometer follows the specification, so the iPhone's measured `accel_sign` correction (−1) is not used (`phone.device: samsung`), see [`setup/windows-android`](../../setup/windows-android/README.md).
