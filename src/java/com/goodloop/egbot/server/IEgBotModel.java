package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.maths.ITrainable;

public interface IEgBotModel extends ITrainable.Unsupervised<Map>, IHasDesc {

	/**
	 * 
	 * @param question
	 * @param possibleAnswer
	 * @return the semantics of this can be whatever -- provided it makes sense for this model,
	 * and that higher = better. 
	 */
	double scoreAnswer(String question, String possibleAnswer) throws IOException;

	/**
	 * Get a sample of words from the model.
	 * 
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	String sample(String question, int expectedAnswerLength) throws IOException;
	
	/**
	 * Get the most likely series of words from the model.
	 *  
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	String generateMostLikely(String question, int expectedAnswerLength) throws IOException;
	
	/**
	 * Internal state model loading if it exists.
	 * 
	 * NB: The overall IEgbotModel object is load/saved by {@link EgBotDataLoader} using Depot.
	 * 
	 * Use-case for this: e.g. TensorFlow checkpoint files.
	 */
	void load() throws IOException;

	/**
	 * initialise any model parameters to prepare for training
	 * @param num_epoch 
	 */
	void init(List<File> files, int num_epoch, String preprocessing, String wordEmbed) throws IOException;

	/** 
	 * used once training finished on all egbot files
	 */
	void setTrainSuccessFlag(boolean b);

	void setLoadSuccessFlag(boolean b);

	String getModelConfig();
}
