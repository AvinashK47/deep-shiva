require('dotenv').config();
const express = require("express");

const app = express();

app.get("/", (req, res) => {
    res.send("yo from root");
});

app.listen(process.env.PORT, () => {
    console.log(`Server running on port ${process.env.PORT}`);
});