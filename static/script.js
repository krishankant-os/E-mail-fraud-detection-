document.getElementById("predictionForm").addEventListener("submit", async (e) => {
e.preventDefault();

const text = document.getElementById("prompt-area").value;
const result = document.getElementById("result");

if (!text.trim()) {
    result.textContent = "Please enter an email.";
    result.className = "error";
    return;
}

try {
    result.textContent = "Checking...";
    result.className = "";

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text
        })
    });

    if (!response.ok) {
        throw new Error("Prediction request failed");
    }

    const data = await response.json();

    result.textContent = `Prediction: ${data.prediction}`;
    result.className = "success";

} catch (error) {
    console.error(error);

    result.textContent = "Something went wrong. Please try again.";
    result.className = "error";
}


});
