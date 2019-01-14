package com.goodloop.egbot.server;

import java.util.List;
import java.util.Map;

import com.winterwell.datascience.Experiment;
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;

public class EgBotExperiment extends Experiment<EgBotData, IEgBotModel, EgBotResults> {

	public EgBotExperiment() {
		// init with blank results
		setResults(new EgBotResults());
	}
	
}

class EgBotResults {
	/**
	 * See {@link QuantModelEvaluator}
	 */
	MeanVar1D avgScore;
	
	/**
	 * See {@link QualModelEvaluator}
	 */
	List<Map<String,?>> generatedAnswers;
}
