let video;
let poseNet;
let poses = [];
let poseHistory = [];
let currentStream;
let usingFrontCamera = false;
let recognizing = false;
let speechRecognizer;
let captionsDiv;
let captionsLines = []; // Array to hold lines of captions
let confidenceLevel = 0.9;
let gainValue = 4.0; // Default gain value
let audioContext;
let gainNode;
let mediaStreamSource;
let analyser;
let dataArray;
let peakValue = 0;
let peakTimestamp = 0;

let maxLines = 14; // Maximum number of lines to display
let videoWidth, videoHeight;
let resizeTimeout;

// Variables to handle rotated keypoints and skeletons
let isPortrait = false;

function setup() {
  pixelDensity(1); // Ensure consistent rendering
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.position(0, 0);
  canvas.style('z-index', '2');

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
  gainSlider.attribute('min', 0.1); // Ensure minimum value is 0.1
  gainSlider.attribute('max', 10); // Set the maximum value of the gain slider to 10
  gainSlider.value(gainValue); // Set the default value to 4
  gainSlider.input(updateGainValue);

  const confirmPopupButton = select('#confirmPopup');
  confirmPopupButton.mousePressed(confirmPopup);

  const closePopupButton = select('#closePopup');
  closePopupButton.mousePressed(hideControlPopup);

  captionsDiv = select('#captions');
  captionsDiv.mousePressed(hideCaptions); // Add this line to hide captions on tap

  // Load settings from local storage
  loadSettings();

  // Enable text recognition by default
  startSpeechRecognition();

  // Hide the control popup initially
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
    // Portrait mode
    isPortrait = true;
    constraints = {
      video: {
        facingMode: usingFrontCamera ? 'user' : 'environment',
        width: { ideal: 720 },
        height: { ideal: 1280 }
      },
      audio: false
    };
  } else {
    // Landscape mode
    isPortrait = false;
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

    // Append the video element to the container
    let videoContainer = document.getElementById('videoContainer');
    if (videoContainer && !videoContainer.contains(video.elt)) {
      videoContainer.appendChild(video.elt);
    }
  });

  video.elt.setAttribute('playsinline', 'true');
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    setupCamera();
  }, 300); // Delay to allow the orientation change to complete
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

  if (isPortrait) {
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

          // Map to canvas coordinates after rotation
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

function isVuMeterVisible() {
  const controlPopup = select('#controlPopup');
  return controlPopup.style('display') !== 'none';
}

function getBaseColor(index) {
  if (index === 0) return color(255, 0, 0);
  let hueValue = (index * 60) % 360;
  return color('hsb(' + hueValue + ', 100%, 50%)');
}

function drawEye(x, y, ageFactor) {
  let eyeColor = lerpColor(color(255), color(0), ageFactor);
  fill(eyeColor);
  stroke(0);
  strokeWeight(1);
  ellipse(x, y, 10, 10);
  fill(0);
  noStroke();
  ellipse(x, y, 3, 3);
}

function drawNose(x, y, ageFactor) {
  let noseColor = lerpColor(color(255, 204, 0), color(0), ageFactor);
  fill(noseColor);
  noStroke();
  ellipse(x, y, 8, 8);
}

function drawEar(x, y, ageFactor) {
  let earColor = lerpColor(color(255, 204, 0), color(0), ageFactor);
  fill(earColor);
  noStroke();
  ellipse(x, y, 8, 16);
}

function toggleCaptions() {
  if (recognizing) {
    speechRecognizer.stop();
    if (audioContext) {
      audioContext.close().then(() => {
        audioContext = null;
        if (mediaStreamSource) {
          mediaStreamSource.mediaStream.getTracks().forEach(track => track.stop());
          mediaStreamSource = null;
        }
      });
    }
    recognizing = false;
    select('#toggleCaptions').html('🤐');
  } else {
    startSpeechRecognition();
    recognizing = true;
    select('#toggleCaptions').html('💬');
  }
}

function startSpeechRecognition() {
  if ('webkitSpeechRecognition' in window) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    gainNode = audioContext.createGain();
    gainNode.gain.value = gainValue; // Use the current gain value

    navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
      mediaStreamSource = audioContext.createMediaStreamSource(stream);
      mediaStreamSource.connect(gainNode);

      // Set up the analyser node
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      gainNode.connect(analyser);
      dataArray = new Uint8Array(analyser.frequencyBinCount);

      // Create a new MediaStream with the adjusted gain node
      const destinationStream = audioContext.createMediaStreamDestination();
      gainNode.connect(destinationStream);

      speechRecognizer = new webkitSpeechRecognition();
      speechRecognizer.continuous = true;
      speechRecognizer.interimResults = true;
      speechRecognizer.lang = 'en-US';
      speechRecognizer.maxAlternatives = 3; // Increased from 1 to 3

      speechRecognizer.onstart = function() {
        recognizing = true;
        select('#toggleCaptions').html('💬');
      };

      speechRecognizer.onend = function() {
        recognizing = false;
        select('#toggleCaptions').html('🤐');
        // Restart recognition to keep listening
        if (recognizing) {
          startSpeechRecognition();
        }
      };

      speechRecognizer.onresult = function(event) {
        let interimTranscript = '';
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + ' ';
          } else {
            interimTranscript += event.results[i][0].transcript + ' ';
          }
        }
        appendCaptions(finalTranscript); // Append the final transcript to captions
        updateCaptions();
      };

      // Start the speech recognition
      speechRecognizer.start();
    }).catch(function(err) {
      console.log('Error accessing the microphone: ' + err);
    });
  } else {
    console.log('Speech recognition not supported.');
  }
}

function appendCaptions(newText) {
  // Split the new text into lines based on the available width
  let words = newText.split(' ');
  let line = '';
  words.forEach(word => {
    let testLine = line + word + ' ';
    let testWidth = textWidth(testLine);
    if (testWidth > width - 20 && line.length > 0) {
      captionsLines.push(line.trim());
      line = word + ' ';
    } else {
      line = testLine;
    }
  });
  if (line.length > 0) {
    captionsLines.push(line.trim());
  }

  // Ensure we only keep the last `maxLines` lines
  if (captionsLines.length > maxLines) {
    captionsLines = captionsLines.slice(captionsLines.length - maxLines);
  }
}

function updateCaptions() {
  captionsDiv.html(''); // Clear current captions
  let opacityStep = 1 / maxLines;

  captionsLines.forEach((line, index) => {
    let lineElement = createP(line);
    lineElement.style('margin', '0');
    lineElement.style('padding', '0');
    lineElement.style('opacity', (1 - (opacityStep * (maxLines - index))).toFixed(2));
    lineElement.style('font-size', '14px');
    lineElement.parent(captionsDiv);
  });

  captionsDiv.style('white-space', 'normal'); // Allow captions to wrap to multiple lines if needed
}

function hideCaptions() {
  captionsDiv.html(''); // Clear current captions
}

function updateSliderValue() {
  let slider = select('#confidenceSlider');
  let newConfidence = slider.value();
  select('#popupConfidenceLevel').html(newConfidence + '%');
}

function updateGainValue() {
  let slider = select('#gainSlider');
  let newGain = slider.value();
  gainValue = parseFloat(newGain);
  select('#popupGainLevel').html(newGain.toFixed(1));
  if (gainNode) {
    gainNode.gain.value = gainValue;
  }
}

function confirmPopup() {
  let slider = select('#confidenceSlider');
  confidenceLevel = slider.value() / 100;
  updateConfidenceLevel();
  saveSettings();
  // No need to reload the page; adjust pose history or other parameters if necessary
}

function updateConfidenceLevel() {
  select('#confidenceLevel').html((confidenceLevel * 100).toFixed(0) + '%');
  select('#popupConfidenceLevel').html((confidenceLevel * 100).toFixed(0) + '%');
}

function toggleControlPopup() {
  let controlPopup = select('#controlPopup');
  if (controlPopup.style('display') === 'none' || controlPopup.style('display') === '') {
    showControlPopup();
  } else {
    hideControlPopup();
  }
}

function showControlPopup() {
  select('#confidenceSlider').value(confidenceLevel * 100);
  select('#popupConfidenceLevel').html((confidenceLevel * 100).toFixed(0) + '%');
  select('#gainSlider').value(gainValue);
  select('#popupGainLevel').html(gainValue.toFixed(1));
  select('#controlPopup').style('display', 'flex');
}

function hideControlPopup() {
  select('#controlPopup').style('display', 'none');
}

function saveSettings() {
  localStorage.setItem('confidenceLevel', confidenceLevel);
  localStorage.setItem('gainValue', gainValue);
}

function loadSettings() {
  const savedConfidenceLevel = localStorage.getItem('confidenceLevel');
  const savedGainValue = localStorage.getItem('gainValue');
  if (savedConfidenceLevel !== null) {
    confidenceLevel = parseFloat(savedConfidenceLevel);
    updateConfidenceLevel();
    select('#confidenceSlider').value(confidenceLevel * 100); // Update the slider value
  }
  if (savedGainValue !== null) {
    gainValue = parseFloat(savedGainValue);
    select('#popupGainLevel').html(gainValue.toFixed(1));
    select('#gainSlider').value(gainValue); // Update the slider value
  } else {
    select('#gainSlider').value(gainValue); // Ensure default gain value is set if no saved value
  }
}

function updateVuMeter() {
  if (analyser && recognizing) {
    analyser.getByteFrequencyData(dataArray);
    let volume = Math.max(...dataArray) / 256;
    let vuMeterFill = select('#audioLevelIndicator');

    if (volume > peakValue) {
      peakValue = volume;
      peakTimestamp = millis();
    }

    let clipping = volume >= 1.0;
    vuMeterFill.style('width', (volume * 100) + '%');
    vuMeterFill.style('background-color', clipping ? 'red' : 'green');

    // Display the peak value
    let peakLine = select('#peakLine');
    if (millis() - peakTimestamp <= 1000) {
      peakLine.style('left', (peakValue * 100) + '%');
      peakLine.style('background-color', 'yellow');
    } else {
      peakValue = 0;
      peakLine.style('left', '0%');
    }
  } else {
    // Set VU meter to zero when mic is turned off
    let vuMeterFill = select('#audioLevelIndicator');
    vuMeterFill.style('width', '0%');
    vuMeterFill.style('background-color', 'green');

    let peakLine = select('#peakLine');
    peakLine.style('left', '0%');
    peakValue = 0;
  }
}
