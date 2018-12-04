package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

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
	 * sample a series of words from the model
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	String sample(String question, int expectedAnswerLength) throws IOException;

	/**
	 * train the model
	 * @throws IOException
	 */
	void train(List<File> files) throws IOException;
	
}
