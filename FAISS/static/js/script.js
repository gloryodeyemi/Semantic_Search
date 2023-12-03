document.addEventListener("DOMContentLoaded", function() {
    const dropdowns = document.querySelectorAll('.dropdown');
    dropdowns.forEach((dropdown) => {
        dropdown.addEventListener('click', (event) => {
        const hiddenInfo = event.currentTarget.querySelector('.hidden-info');
            if (hiddenInfo) {
                hiddenInfo.classList.toggle('show-info');
            }
        });
    });
});

var voiceSearchButton = document.getElementById('voice-search-btn');

voiceSearchButton.addEventListener('click', function(event) {
    event.preventDefault();
    startDictation();
});

function startDictation() {
    console.log("startDictation function is being called.");
    if (!('webkitSpeechRecognition' in window)) {
        upgrade();
    } else {
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;

        recognition.onstart = function() {
            console.log("Recognition started.");
            voiceSearchButton.style.backgroundColor = "#ff0000"; // red color to indicate active listening
            voiceSearchButton.innerHTML = "Listening..."; // change the button text to indicate active listening
        };

        recognition.onresult = function(event) {
            console.log("Recognition results: ", event);
            const query = event.results[0][0].transcript; // get the recognized text
            document.getElementById('query-text').value = query; // set the query in the input field
        };

        recognition.onerror = function(event) {
            console.error("Recognition error: ", event);
            if (event.error === 'not-allowed') {
                console.log("Microphone access not allowed. Please enable the microphone.");
                alert("Microphone access is not allowed. Please enable the microphone in your browser settings.");
            }
            recognition.stop();
        };

        recognition.onend = function() {
            console.log("Recognition ended.");
            voiceSearchButton.innerHTML = "Voice search";
            voiceSearchButton.style.backgroundColor = "#009688";
            recognition.stop();
        };

        // code for the start button
        recognition.lang = "en-US";
        recognition.start();
    }
}