const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        detectEmotion();
    })
    .catch((error) => {
        console.error('Error accessing webcam:', error);
    });

function detectEmotion() {
    const interval = setInterval(() => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        // Implement emotion detection logic using imageData

        // Example: send imageData to Flask backend for processing
        // const data = { image_data: imageData };
        // fetch('/detect_emotion', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify(data),
        // })
        // .then(response => response.json())
        // .then(result => console.log(result))
        // .catch(error => console.error('Error detecting emotion:', error));
    }, 1000);  // Adjust the interval as needed
}
