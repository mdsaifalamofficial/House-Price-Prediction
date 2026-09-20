const form =
    document.getElementById(
        "predictionForm"
    );


const result =
    document.getElementById(
        "result"
    );


const price =
    document.getElementById(
        "price"
    );


const error =
    document.getElementById(
        "error"
    );


form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        // Hide previous messages

        result.classList.add(
            "hidden"
        );

        error.classList.add(
            "hidden"
        );


        // Get form values

        const location =
            document.getElementById(
                "location"
            ).value;


        const size =
            document.getElementById(
                "size"
            ).value;


        const rooms =
            document.getElementById(
                "rooms"
            ).value;


        const age =
            document.getElementById(
                "age"
            ).value;


        // Create request body

        const requestData = {

            location: location,

            size: Number(size),

            rooms: Number(rooms),

            age: Number(age)

        };


        try {

            // Send request to FastAPI

            const response =
                await fetch(
                    "https://house-price-prediction-api-kura.onrender.com/",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                requestData
                            )
                    }
                );


            // Check API response

            if (!response.ok) {

                throw new Error(
                    "Prediction request failed."
                );

            }


            // Convert JSON response

            const data =
                await response.json();


            // Display price

            price.textContent =
                `₹${Number(
                    data.predicted_price
                ).toLocaleString(
                    "en-IN",
                    {
                        maximumFractionDigits: 0
                    }
                )}`;


            result.classList.remove(
                "hidden"
            );

        }

        catch (err) {

            error.textContent =
                "Unable to get prediction. Please make sure the FastAPI server is running.";

            error.classList.remove(
                "hidden"
            );

            console.error(err);

        }

    }
);