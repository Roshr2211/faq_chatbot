const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const dotenv = require('dotenv');
const path = require('path');

const cors = require('cors');
dotenv.config();

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use(cors());

app.post('/chat', (req, res) => {
    const { question } = req.body;
    
    if (!question) {
        return res.status(400).json({ error: 'Question is required' });
    }

    console.log(`Received question: ${question}`);

    const pythonProcess = spawn('python', [path.join(__dirname, 'langchain_faq.py'), question]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python script output: ${data.toString()}`);
        try {
            const response = JSON.parse(data.toString());
            res.json(response);
        } catch (e) {
            console.error('Error parsing JSON from Python script:', e);
            res.status(500).json({ error: 'Failed to parse answer from chatbot' });
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python script error: ${data.toString()}`);
        res.status(500).json({ error: 'Failed to get an answer from the chatbot' });
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python script exited with code ${code}`);
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
