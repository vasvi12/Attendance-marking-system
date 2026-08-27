/**
 * Shared front-end helpers: mobile nav toggle, CSRF-aware fetch, and small
 * camera utilities reused by the enrollment (capture_faces.html) and
 * live-attendance (live_attendance.html) pages.
 */

(function () {
  "use strict";

  // -- Mobile nav toggle ----------------------------------------------------

  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });
  }

  // -- CSRF-aware fetch -------------------------------------------------------

  function getCookie(name) {
    const match = document.cookie.match(
      new RegExp("(^|;\\s*)" + name + "=([^;]*)")
    );
    return match ? decodeURIComponent(match[2]) : null;
  }

  /**
   * POST a JSON-serializable object to `url`, including the Django CSRF
   * token, and return the parsed JSON response.
   */
  window.postJSON = async function postJSON(url, data) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.message || detail;
      } catch (err) {
        /* response wasn't JSON; keep statusText */
      }
      throw new Error(`Request failed (${response.status}): ${detail}`);
    }
    return response.json();
  };

  // -- Camera helpers -----------------------------------------------------

  /**
   * Request the user's webcam and attach it to a <video> element.
   * Returns the MediaStream so the caller can stop it later. Throws a
   * descriptive error the UI can display if the camera is unavailable or
   * permission is denied.
   */
  window.startCamera = async function startCamera(videoEl) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser does not support camera access.");
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (err) {
      if (err.name === "NotAllowedError") {
        throw new Error("Camera permission was denied. Allow camera access and reload.");
      }
      if (err.name === "NotFoundError") {
        throw new Error("No camera was found on this device.");
      }
      throw new Error("Could not access the camera: " + err.message);
    }
    videoEl.srcObject = stream;
    await videoEl.play();
    return stream;
  };

  window.stopCamera = function stopCamera(stream) {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  };

  /**
   * Draw the current video frame onto a hidden canvas and return it as a
   * base64 JPEG data URL, ready to POST to the backend.
   */
  window.captureFrameDataURL = function captureFrameDataURL(videoEl, quality) {
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality || 0.85);
  };
})();
