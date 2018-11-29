package com.goodloop.egbot.server;

import java.util.Map;

import com.winterwell.maths.ITrainable;

public interface IEgBotModel extends ITrainable.Unsupervised<Map> {

	/**
	 * 
	 * @param question
	 * @param possibleAnswer
	 * @return the semantics of this can be whatever -- provided it makes sense for this model,
	 * and that higher = better. 
	 */
	double scoreAnswer(String question, String possibleAnswer);

	String sampleSeries(String question, int expectedAnswerLength);

}
