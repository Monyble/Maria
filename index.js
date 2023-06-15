const tf = require('@tensorflow/tfjs-node');
const express = require('express');
const startModel = require('./initiateModelTraining');

const app = express();
// Load the saved model
const MODEL_PATH = 'gan_model/model.json';

app.get('/generate', async (req, res) => {
    const prompt = req.query.prompt;
  
    try {
      // Load the trained model
      const model = await tf.loadLayersModel(`file://${MODEL_PATH}`);
  
      // Preprocess the prompt if needed
      const preprocessedPrompt = preprocessText(prompt);
  
      // Pad the prompt to the desired length
      const paddedPrompt = padPrompt(preprocessedPrompt, 100);
  
      // Convert the padded prompt to numeric values
      const input = tf.tensor2d([paddedPrompt.split('').map(char => char.charCodeAt(0))], [1, 100]);
  console.log(input)
      // Generate the answer using the model
      const output = model.predict(input);
  console.log(output)
      // Decode and send the generated answer
      const generatedText = decodeText(output);
      res.send(generatedText);
    } catch (error) {
      console.error('Error generating answer:', error);
      res.status(500).send('Error generating answer');
    }
  });
  
  // Preprocess the text (lowercase, remove punctuation, etc.)
  function preprocessText(text) {
    // Implement your preprocessing logic here
    // For example:
    const preprocessedText = text.toLowerCase().replace(/[^\w\s]/g, '');
    return preprocessedText;
  }
  
  // Pad the prompt to a specified length
  function padPrompt(prompt, length) {
    if (prompt.length >= length) {
      return prompt.slice(0, length);
    } else {
      return prompt.padEnd(length, ' ');
    }
  }
  
  
  // Decode the generated text from the model's output
  function decodeText(output) {
    // Implement your decoding logic here
    // Convert the tensor output to a readable string
    const decodedText = output.dataSync().toString();
    return decodedText;
  }
  

setInterval(() => {
    
    startModel();
}, 1000);
  
  
  
  
  

// Start the server
app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
