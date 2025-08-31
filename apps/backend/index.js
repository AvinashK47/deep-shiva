require("dotenv").config();
const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Yo from backend root");
});

app.post("/chat", async (req, res) => {
  try {
    
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }
      
    const aiResponse = await axios.post("http://localhost:8000/api/chat", {
      query: query, 
    });
    res.json(aiResponse.data);
    
  } catch (error) {
      // the AI server is down
    console.error("Error communicating with AI server:", error.message);
    res.status(500).json({ error: "Failed to get response from AI server" });
  }
});

app.listen(process.env.PORT || 3000, () => {
  console.log(`Backend server running on port ${process.env.PORT || 3000}`);
});