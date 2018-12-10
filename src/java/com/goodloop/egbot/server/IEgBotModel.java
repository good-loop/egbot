package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import com.winterwell.depot.IHasDesc;
import com.winterwell.maths.ITrainable;
import com.winterwell.nlp.io.SitnStream;

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
	 * Get the most likely series of words from the model.
	 * 
	 * TODO rename from `sample`
	 * 
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	String sample(String question, int expectedAnswerLength) throws IOException;
	
	/**
	 * model loading if it exists
	 */
	void load() throws IOException;

	/**
	 * get model object
	 * @return
	 */
	Object getWmc();
	
	/**
	 * initialise any model parameters to prepare for training
	 */
	void init(List<File> files) throws IOException;	
}
