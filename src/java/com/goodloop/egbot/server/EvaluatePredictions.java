package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import com.winterwell.depot.Depot;
import com.winterwell.depot.DepotConfig;
import com.winterwell.depot.Desc;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.log.LogFile;
/**
 * To evaluate EgBot!
 * 
 * First run {@link ConstructEvaluationSet}
 * 
 * Then run this.
 * 
 * @author Irina
 * @testedby {@link EvaluationTest}
 */
public class EvaluatePredictions {
	
	public static void main(String[] args) throws Exception {
		new EvaluatePredictions().run();
	}
	
	public EvaluatePredictions() {
	}
	
	static LogFile logFile = new LogFile();
	
	public void run() throws Exception {
			
		// Markov 
		MarkovModel mm = new MarkovModel("MSE-20");		
		mm.load();
		runModel(mm, "MSE-20", "MSE-20");		
				
		// LSTM 
//		LSTM lstm = new LSTM();				
//		runModel(lstm);				
	}
	
	/**
	 * train model and then run evaluations (quant and qual)
	 * 
	 * @param model
	 * @param tLabel training data label (e.g. "MSE-full", "MSE-part", "MSE-20", "paul-20")
	 * @param eLabel evaluation data label
	 * @throws Exception
	 */
	void runModel(IEgBotModel model, String tLabel, String eLabel) throws Exception {
		Desc<IEgBotModel> modelDesc = model.getDesc();
		//		modelDesc.put("trainingFiles", "MSE-part");
		
		// refresh cache?
		//Depot.getDefault().remove(modelDesc);
		
		// TRAIN
		
		// decide on train data version (e.g. MSE-full, MSE-20, paul-20)
		String trainLabel = tLabel;
		Log.i("Using following training data: " + trainLabel);
		// load the list of files
		List<File> files = EgBotDataLoader.setup(trainLabel);

		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 2 != 1;
		IFilter<Integer> testFilter = n -> ! trainFilter.accept(n);

		// set up experiment
		EgBotExperiment experiment = trainExp(model, modelDesc, trainFilter, files, trainLabel);
		
		// TEST
		
		// decide on eval data version (e.g. MSE-full, MSE-20, paul-20)
		String evalLabel = eLabel;
		Log.i("Using following eval data: " + evalLabel);
		// load eval files (e.g. Paulius' 20 questions set)
		List<File> evalFiles = EgBotDataLoader.setup(evalLabel);
		// set the test filter		
		EgBotData testData = new EgBotData(evalFiles, testFilter);
		// set the test data the experiment uses
		Desc testDataDesc = new Desc(evalLabel, EgBotData.class);
		testDataDesc.put("use", "test");
		experiment.setTestData(testData, testDataDesc);

		if (true) { // QUANT EVAL
			// set up quantitative evaluator
			QuantModelEvaluator quant = new QuantModelEvaluator(experiment);
			// conduct evaluation
			quant.evaluateModel();
		}
		
		if (true) {//QUAL EVAL			
			// set up qualitative evaluator
			QualModelEvaluator qual = new QualModelEvaluator(experiment);
			// conduct evaluation
			qual.evaluateModel();
		}
		
		// NB: the evaluator classes both save results
		Log.i("Trained model at: "+Depot.getDefault().getLocalPath(experiment.getModel().getDesc()));
		Log.i("Model guts at: "+Depot.getDefault().getLocalPath(experiment.getModel().getWmcDesc()));
		Log.i("Results at: "+Depot.getDefault().getLocalPath(experiment.getDesc()));
	}
	
	/**
	 * 
	 * @param model
	 * @param modelDesc
	 * @param trainFilter
	 * @param testFilter
	 * @param files Training files
	 * @return
	 * @throws IOException
	 */
	public EgBotExperiment trainExp(IEgBotModel model, Desc<IEgBotModel> modelDesc, 
			IFilter<Integer> trainFilter,  
			List<File> files, String trainLabel) throws IOException 
	{
		// set up new experiment
		EgBotExperiment experiment = new EgBotExperiment();
		
		// get experiment desc? 
		//Desc expDesc = experiment.getDesc();
		
		// set the model the experiment uses
		experiment.setModel(model, modelDesc);

		// Train
		// set the train filter		
		EgBotData trainData = new EgBotData(files, trainFilter);
		// set the train data the experiment uses
		Desc trainDataDesc = new Desc(trainLabel, EgBotData.class);
		trainDataDesc.put("use", "train");		
		experiment.setTrainData(trainData, trainDataDesc);
		// already trained?
		IEgBotModel trainedModel = Depot.getDefault().get(modelDesc);
		if (trainedModel==null) {
			// do training
			EgBotDataLoader.train(experiment);
			Depot.getDefault().put(experiment.getModel().getDesc(), experiment.getModel());
		} else {
			// replace the untrained with the trained
			Log.d("Using pre-trained model");
			experiment.setModel(trainedModel, modelDesc);
		}		
		return experiment;
	}
	
}
