let video;
let poseNet;
let poses = [];
let poseHistory = [];
let currentStream;
let usingFrontCamera = false;
let recognizing = false;
let speechRecognizer;
let captionsDiv;
let captionsLines = [];
let confidenceLevel = 0.9;
let gainValue = 4.0;
let audioContext;
let gainNode;
let mediaStreamSource;
let analyser;
let dataArray;
let peakValue = 0;
let peakTimestamp = 0;
let maxLines = 14;
let videoWidth, videoHeight;

function setup() {
  pixelDensity(1); // Ensure consistent rendering
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.position(0, 0);

  setupCamera();

  // Initialize UI elements
  const confidenceDisplay = select('#confidenceLevel');
  confidenceDisplay.mousePressed(toggleControlPopup);

  const switchButton = select('#switchCamera');
  switchButton.mousePressed(switchCamera);

  const toggleCaptionsButton = select('#toggleCaptions');
  toggleCaptionsButton.mousePressed(toggleCaptions);

  const confidenceSlider = select('#confidenceSlider');
  confidenceSlider.input(updateSliderValue);

  const gainSlider = select('#gainSlider');
  gainSlider.attribute('min', 0.1);
  gainSlider.attribute('max', 10);
  gainSlider.value(gainValue);
  gainSlider.input(updateGainValue);

  const confirmPopupButton = select('#confirmPopup');
  confirmPopupButton.mousePressed(confirmPopup);

  const closePopupButton = select('#closePopup');
  closePopupButton.mousePressed(hideControlPopup);

  captionsDiv = select('#captions');
  captionsDiv.mousePressed(hideCaptions);

  loadSettings();
  startSpeechRecognition();
  hideControlPopup();

  // Handle window resize and orientation change
  windowResized();
}

function setupCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  let constraints;

  if (window.innerHeight > window.innerWidth) {
    // Portrait mode: Higher height than width
    constraints = {
      video: {
        facingMode: usingFrontCamera ? 'user' : 'environment',
        width: { ideal: 720 },
        height: { ideal: 1280 }
      },
      audio: false
    };
  } else {
    // Landscape mode: Higher width than height
    constraints = {
      video: {
        facingMode: usingFrontCamera ? 'user' : 'environment',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };
  }

  video = createCapture(constraints, function(stream) {
    currentStream = stream;
    video.size(windowWidth, windowHeight);
    video.hide();

    poseNet = ml5.poseNet(video, {
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: 256,
      multiplier: 0.75
    }, modelReady);

    poseNet.on("pose", function(results) {
      poses = results;
      poseHistory.push({poses: results, timestamp: millis()});
      poseHistory = poseHistory.filter(entry => millis() - entry.timestamp <= 2000);
    });

    document.getElementById('videoContainer').appendChild(video.elt);
  });

  video.elt.setAttribute('playsinline', 'true');
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  setupCamera();
}

function switchCamera() {
  usingFrontCamera = !usingFrontCamera;
  setupCamera();
}

function modelReady() {
  console.log("PoseNet Model Ready");
}

function draw() {
  background(0);

  let videoAspect = video.width / video.height;
  let canvasAspect = width / height;

  if (window.innerHeight > window.innerWidth) {
    // Portrait Mode: Rotate the canvas
    push();
    translate(width / 2, height / 2);
    rotate(PI / 2); // Rotate 90 degrees

    // After rotation, width and height are swapped
    if (canvasAspect > videoAspect) {
      videoHeight = width * videoAspect;
      videoWidth = width;
    } else {
      videoWidth = height / videoAspect;
      videoHeight = height;
    }

    // Draw the video with swapped dimensions
    image(video, -videoWidth / 2, -videoHeight / 2, videoWidth, videoHeight);
    pop();

    // Draw keypoints and skeletons adjusted for rotation
    drawKeypointsRotated();
    drawSkeletonsRotated();
  } else {
    // Landscape Mode: Draw normally
    if (canvasAspect > videoAspect) {
      videoWidth = height * videoAspect;
      videoHeight = height;
    } else {
      videoWidth = width;
      videoHeight = width / videoAspect;
    }

    let x = (width - videoWidth) / 2;
    let y = (height - videoHeight) / 2;

    image(video, x, y, videoWidth, videoHeight);

    // Draw keypoints and skeletons normally
    drawKeypoints(x, y, videoWidth, videoHeight);
    drawSkeletons(x, y, videoWidth, videoHeight);
  }

  if (isVuMeterVisible()) {
    updateVuMeter();
  }
}

function drawKeypoints(xOffset, yOffset, videoWidth, videoHeight) {
  for (let historyEntry of poseHistory) {
    let ageFactor = (millis() - historyEntry.timestamp) / 2000;
    for (let i = 0; i < historyEntry.poses.length; i++) {
      const pose = historyEntry.poses[i].pose;
      for (let j = 0; j < pose.keypoints.length; j++) {
        const keypoint = pose.keypoints[j];
        if (keypoint.score > confidenceLevel) {
          let x = map(keypoint.position.x, 0, video.width, xOffset, xOffset + videoWidth);
          let y = map(keypoint.position.y, 0, video.height, yOffset, yOffset + videoHeight);
          if (keypoint.part === 'leftEye' || keypoint.part === 'rightEye') {
            drawEye(x, y, ageFactor);
          } else if (keypoint.part === 'nose') {
            drawNose(x, y, ageFactor);
          } else if (keypoint.part === 'leftEar' || keypoint.part === 'rightEar') {
            drawEar(x, y, ageFactor);
          } else {
            fill(255, 0, 0);
            noStroke();
            ellipse(x, y, 5, 5);
          }
        }
      }
    }
  }
}

function drawKeypointsRotated() {
  for (let historyEntry of poseHistory) {
    let ageFactor = (millis() - historyEntry.timestamp) / 2000;
    for (let i = 0; i < historyEntry.poses.length; i++) {
      const pose = historyEntry.poses[i].pose;
      for (let j = 0; j < pose.keypoints.length; j++) {
        const keypoint = pose.keypoints[j];
        if (keypoint.score > confidenceLevel) {
          // Rotate keypoint positions by 90 degrees
          let rotatedX = keypoint.position.y;
          let rotatedY = video.width - keypoint.position.x;
          
          // Map to canvas coordinates
          rotatedX = map(rotatedX, 0, video.width, (height - videoHeight) / 2, (height + videoHeight) / 2);
          rotatedY = map(rotatedY, 0, video.height, (width - videoWidth) / 2, (width + videoWidth) / 2);

          if (keypoint.part === 'leftEye' || keypoint.part === 'rightEye') {
            drawEye(rotatedX, rotatedY, ageFactor);
          } else if (keypoint.part === 'nose') {
            drawNose(rotatedX, rotatedY, ageFactor);
          } else if (keypoint.part === 'leftEar' || keypoint.part === 'rightEar') {
            drawEar(rotatedX, rotatedY, ageFactor);
          } else {
            fill(255, 0, 0);
            noStroke();
            ellipse(rotatedX, rotatedY, 5, 5);
          }
        }
      }
    }
  }
}

function drawSkeletons(xOffset, yOffset, videoWidth, videoHeight) {
  for (let historyEntry of poseHistory) {
    let ageFactor = (millis() - historyEntry.timestamp) / 2000;
    for (let i = 0; i < historyEntry.poses.length; i++) {
      const skeleton = historyEntry.poses[i].skeleton;
      let baseColor = getBaseColor(i);
      for (let j = 0; j < skeleton.length; j++) {
        const partA = skeleton[j][0];
        const partB = skeleton[j][1];
        let x1 = map(partA.position.x, 0, video.width, xOffset, xOffset + videoWidth);
        let y1 = map(partA.position.y, 0, video.height, yOffset, yOffset + videoHeight);
        let x2 = map(partB.position.x, 0, video.width, xOffset, xOffset + videoWidth);
        let y2 = map(partB.position.y, 0, video.height, yOffset, yOffset + videoHeight);
        stroke(lerpColor(baseColor, color(0, 0, 0), ageFactor));
        line(x1, y1, x2, y2);
      }
    }
  }
}

function drawSkeletonsRotated() {
  for (let historyEntry of poseHistory) {
    let ageFactor = (millis() - historyEntry.timestamp) / 2000;
    for (let i = 0; i < historyEntry.poses.length; i++) {
      const skeleton = historyEntry.poses[i].skeleton;
      let baseColor = getBaseColor(i);
      for (let j = 0; j < skeleton.length; j++) {
        const partA = skeleton[j][0];
        const partB = skeleton[j][1];

        // Rotate partA
        let rotatedAX = partA.position.y;
        let rotatedAY = video.width - partA.position.x;
        rotatedAX = map(rotatedAX, 0, video.width, (height - videoHeight) / 2, (height + videoHeight) / 2);
        rotatedAY = map(rotatedAY, 0, video.height, (width - videoWidth) / 2, (width + videoWidth) / 2);

        // Rotate partB
        let rotatedBX = partB.position.y;
        let rotatedBY = video.width - partB.position.x;
        rotatedBX = map(rotatedBX, 0, video.width, (height - videoHeight) / 2, (height + videoHeight) / 2);
        rotatedBY = map(rotatedBY, 0, video.height, (width - videoWidth) / 2, (width + videoWidth) / 2);

        stroke(lerpColor(baseColor, color(0, 0, 0), ageFactor));
        line(rotatedAX, rotatedAY, rotatedBX, rotatedBY);
      }
    }
  }
}

// ... [Rest of your existing functions remain unchanged] ...
