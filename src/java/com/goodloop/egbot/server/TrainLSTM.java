package com.goodloop.egbot.server;

import java.util.List;
import java.util.Map;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

public class TrainLSTM {
	
	static void train() {		
		// Java TensorFlow documentation: https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary
		
		// TODO: turn below python Keras code into Java TF code
		//model = Sequential()
		//model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
		//model.add(Dropout(0.6))
		//model.add(Dense(vocab_size))
		//model.add(Activation('softmax'))
		//
		//optimizer = Adam(lr=learning_rate)
		//callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
		//model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
	}
	
	static void sample() throws Exception {
		 String modelPath = "\\data\\models\\v3\\latest.pb";
		 try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
			 printSignature(model);
			 String question = "what is a probability";
			 List<Tensor<?>> outputs = null;
			 try (Tensor<UInt8> questionTensor = makeTensor(question)) {
				 outputs = model
				          .session()
				          .runner()
				          .feed("question", questionTensor)
				          .fetch("answer")
				          .run();
			}
			try (Tensor<String> answer = outputs.get(0).expect(String.class);) {
				// TODO convert tensor to string
			}			 
		 }
	}
	
	private static Tensor<UInt8> makeTensor(String input) {
		// TODO return words (could be question or answer) in the form of tensor
		return null;
	}

	/*
	 * Prints out the signature of the model, which specifies what type of model is being exported, 
	 * and the input/output tensors to bind to when running inference.
	 */
	private static void printSignature(SavedModelBundle model) throws Exception {
		MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
		SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
		int numInputs = sig.getInputsCount();
		int i = 1;
		System.out.println("MODEL SIGNATURE");
		System.out.println("Inputs:");
		for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
		  TensorInfo t = entry.getValue();
		  System.out.printf(
		      "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
		      i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
		}
		int numOutputs = sig.getOutputsCount();
		i = 1;
		System.out.println("Outputs:");
		for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
		  TensorInfo t = entry.getValue();
		  System.out.printf(
		      "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
		      i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
		}
		System.out.println("-----------------------------------------------");
	}
}
