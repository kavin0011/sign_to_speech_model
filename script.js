const video = document.getElementById("camera");
const startRecordingBtn = document.getElementById("startRecording");
const stopRecordingBtn = document.getElementById("stopRecording");
const frameCanvas = document.getElementById("frameCanvas");
const outputDiv = document.getElementById("output");
let mediaRecorder, recordedChunks = [], session;

// Define the ONNX model path
const modelPath = "D:/SIH2024/Sign_to_text/best.onnx"; // Replace with actual path
const classNames = ["Hello", "Thank You", "Yes", "No", "I Love You"]; // Replace with your model's classes

// Load the ONNX model
async function loadModel() {
  session = await ort.InferenceSession.create(modelPath);
  console.log("ONNX model loaded successfully.");
  outputDiv.innerText = "Model loaded. Ready to detect sign language.";
}

// Initialize the webcam feed
async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
    console.log("Webcam initialized.");
  } catch (err) {
    outputDiv.innerText = "Failed to access webcam. Check permissions.";
    console.error(err);
  }
}

// Start recording video
function startRecording() {
  recordedChunks = [];
  const stream = video.srcObject;
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = async () => {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    processVideo(blob);
  };
  mediaRecorder.start();
  startRecordingBtn.disabled = true;
  stopRecordingBtn.disabled = false;
  outputDiv.innerText = "Recording...";
}

// Stop recording video
function stopRecording() {
  mediaRecorder.stop();
  startRecordingBtn.disabled = false;
  stopRecordingBtn.disabled = true;
  outputDiv.innerText = "Processing video...";
}

// Process the recorded video and send frames to ONNX model
async function processVideo(blob) {
  const videoElement = document.createElement("video");
  videoElement.src = URL.createObjectURL(blob);
  videoElement.muted = true;
  videoElement.playsinline = true;
  videoElement.onloadeddata = async () => {
    const ctx = frameCanvas.getContext("2d");
    frameCanvas.width = videoElement.videoWidth;
    frameCanvas.height = videoElement.videoHeight;

    videoElement.play();

    while (!videoElement.ended) {
      ctx.drawImage(videoElement, 0, 0, frameCanvas.width, frameCanvas.height);

      const imageData = ctx.getImageData(0, 0, frameCanvas.width, frameCanvas.height);
      const inputTensor = preprocessImage(imageData);

      const results = await session.run({ input: inputTensor });
      const predictions = postprocessResults(results);

      outputDiv.innerText = `Detected: ${predictions}`;
      await new Promise((resolve) => setTimeout(resolve, 500)); // Process every 500ms
    }

    outputDiv.innerText += " - Video processing complete.";
  };
}

// Preprocess image for ONNX model
function preprocessImage(imageData) {
  const data = new Float32Array(imageData.data.length / 4 * 3);
  for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
    data[j] = imageData.data[i] / 255.0;       // R
    data[j + 1] = imageData.data[i + 1] / 255.0; // G
    data[j + 2] = imageData.data[i + 2] / 255.0; // B
  }
  return new ort.Tensor("float32", data, [1, 3, 640, 640]); // Adjust dimensions as per model requirements
}

// Postprocess results to map model output to class names
function postprocessResults(results) {
  const output = results.output; // Adjust key based on your model's output name
  const data = output.data;
  const maxIndex = data.indexOf(Math.max(...data));
  return classNames[maxIndex] || "Unknown";
}

// Initialize everything
(async function main() {
  await loadModel();
  await setupWebcam();
})();

// Event listeners for buttons
startRecordingBtn.addEventListener("click", startRecording);
stopRecordingBtn.addEventListener("click", stopRecording);
