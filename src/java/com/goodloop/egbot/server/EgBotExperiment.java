package com.goodloop.egbot.server;

import java.util.List;
import java.util.Map;

import com.winterwell.datascience.Experiment;
import com.winterwell.depot.Desc;
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;

public class EgBotExperiment extends Experiment<EgBotData, IEgBotModel, EgBotResults> {

	transient IEgBotModel model;
	Desc<IEgBotModel> modelDesc;
	
	EgBotResults results;	
	
	transient EgBotData testData;	
	Desc<EgBotData> testDataDesc;
	
	transient EgBotData trainData;
	Desc<EgBotData> trainDataDesc;
	
	private String tag = "experiment";	
	
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
	
	public List<Map<String, ?>> getQualAnswers() {
		return this.generatedAnswers;
	}
	
	public MeanVar1D getAvgScore() {
		return this.avgScore;
	}
	
	public void setGeneratedAnswers(List<Map<String,?>> generatedAnswers) {
		this.generatedAnswers = generatedAnswers;
	}
	
	public void setAvgScore(MeanVar1D score) {
		this.avgScore = score;
	}

}
