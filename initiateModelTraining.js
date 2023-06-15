const tf = require('@tensorflow/tfjs-node');
const trainingData = require('./assets/trainingData');

// Define the GAN model architecture
// Define the GAN model architecture using the functional API
function defineGANModel() {
  const noiseSize = 100; // Size of the input noise vector
  const numChannels = 3; // Number of color channels in the image

  // Define the generator model
  const generatorInput = tf.input({ shape: [noiseSize] });
  const generatorOutput = tf.layers.dense({ units: numChannels, activation: 'relu' }).apply(generatorInput);
  const generator = tf.model({ inputs: generatorInput, outputs: generatorOutput });

  // Define the discriminator model
  const discriminatorInput = tf.input({ shape: [numChannels] });
  const discriminatorOutput = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(discriminatorInput);
  const discriminator = tf.model({ inputs: discriminatorInput, outputs: discriminatorOutput });

  // Combine the generator and discriminator into a GAN model
  const ganInput = generatorInput;
  const generatedText = generator.apply(ganInput);
  const ganOutput = discriminator.apply(generatedText);
  const gan = tf.model({ inputs: ganInput, outputs: ganOutput });

  return { gan, generator, discriminator };
}

// Train the GAN model
async function trainGANModel(gan, generator, discriminator, data) {
    const numEpochs = 1000; // Specify the number of training epochs
    const batchSize = 32; // Specify the batch size for training
    const learningRate = 0.001; // Specify the learning rate for the optimizer
    const noiseSize = 100; // Specify the size of the input noise vector
    const ganOptimizer = tf.train.adam(learningRate);
    const generatorOptimizer = tf.train.adam(learningRate);
    const discriminatorOptimizer = tf.train.adam(learningRate);
  
     // Compile the models before training
  gan.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
  });
  generator.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
  });
  discriminator.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
    
    for (let epoch = 1; epoch <= numEpochs; epoch++) {
      // Perform one epoch of training
      for (let i = 0; i < data.length; i += batchSize) {
        const realDataBatch = data.slice(i, i + batchSize);
        const noise = tf.randomNormal([batchSize, noiseSize]);
  
        // Train the discriminator
        tf.tidy(() => {
          const generatedDataBatch = generator.predict(noise);
          const realLabels = tf.ones([batchSize, 1]);
          const fakeLabels = tf.zeros([batchSize, 1]);
  
          const discriminatorLossReal = discriminator.trainOnBatch(realDataBatch, realLabels);
          const discriminatorLossFake = discriminator.trainOnBatch(generatedDataBatch, fakeLabels);
          const discriminatorLoss = discriminatorLossReal.add(discriminatorLossFake).div(tf.scalar(2));
        });
  
        // Train the generator
        tf.tidy(() => {
          const generatedDataBatch = generator.predict(noise);
          const ganLabels = tf.ones([batchSize, 1]);
  
          const ganLoss = gan.trainOnBatch(noise, ganLabels);
        });
  
        tf.dispose([realDataBatch, noise]);
      }
  
      console.log(`Epoch ${epoch}/${numEpochs} completed.`);
    }
  
    // Save the trained models if needed
    await gan.save('file://gan_model');
    await generator.save('file://generator_model');
    await discriminator.save('file://discriminator_model');
  }
  

async function preprocessText(text) {
  // Perform any necessary preprocessing on the text
  // Return the preprocessed text

  // Example preprocessing: lowercase and remove punctuation
  const preprocessedText = text.toLowerCase().replace(/[^\w\s]/g, '');

  return preprocessedText;
}

async function loadTrainingData() {
    const data = [];
  
    for (const sample of trainingData) {
      try {
        const preprocessedText = await preprocessText(sample);
  
        // Encode the preprocessed text as integers
        const textEncoder = new TextEncoder();
        const textEncoded = textEncoder.encode(preprocessedText);
  
        // Create a tensor from the encoded text
        const textTensor = tf.tensor(textEncoded, [textEncoded.length, 1], 'float32');
  
        data.push(textTensor);
      } catch (error) {
        console.error(`Error processing sample: ${sample}`);
        console.error(error);
      }
    }
  
    // Concatenate the text tensors along a new axis
    const concatenatedData = tf.concat(data, 0);
  
    return concatenatedData;
  }
  

async function startModel() {
  const noiseSize = 100; // Size of the input noise vector
  const data = await loadTrainingData(); // Load and preprocess the training data
  const { gan, generator, discriminator } = defineGANModel();
  await trainGANModel(gan, generator, discriminator, data);
  console.log('GAN model training complete!');
}



module.exports=startModel