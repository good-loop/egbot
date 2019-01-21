package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import com.winterwell.depot.Depot;
import com.winterwell.depot.DepotConfig;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.maths.ITrainable.Supervised;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.nlp.io.Tkn;
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
	
	@Deprecated
	public void run() throws Exception {
			
		// Markov 
		MarkovModel mm = new MarkovModel();		
		runModel(mm, "MSE-20", "MSE-20", 100, 1);		
				
		//	LSTM 
		//	LSTM lstm = new LSTM();				
		//	runModel(lstm);				
	}
	
	/**
	 * train model and then run evaluations (quant and qual)
	 * 
	 * @param model
	 * @param tLabel training data label (e.g. "MSE-full", "MSE-part", "MSE-20", "paul-20")
	 * @param eLabel evaluation data label (same as above)
	 * TODO: implement filter split param  
	 * @param tSplit percentage of train split (e.g. 2 for the first 1 of 2, 100 for the first 99 of 100), if it's 1 then it's the whole data set 
	 * @param eSplit percentage of train split (e.g. 2 for 1 of 2, 1 etc)
	 * @throws Exception
	 */
	void runModel(IEgBotModel model, String tLabel, String eLabel, int tFilter, int eFilter) throws Exception {
		// TODO: does the lstm have this mm/ wmc equivalence? i suppose there's the LSTM model class and there's the session object
		
		Desc<IEgBotModel> modelDesc = model.getDesc();
		modelDesc.put("train", tLabel);
		
		// refresh cache?
		//Depot.getDefault().remove(modelDesc);
		
		// TRAIN
		
		// decide on train data version (e.g. MSE-full, MSE-20, paul-20)
		Log.i("Using following training data: " + tLabel);
		// load the list of files
		List<File> files = EgBotDataLoader.setup(tLabel);

		// set up filters (that decide train/test split)
		// NB: we specify the filter by passing a parameter x that specifies the train/test split n % x != 1, 
		// if x is 1 then it's the whole dataset (which allows us to use completely different datasets)
		// TODO: check w/ DW that this makes sense in terms of usability
		IFilter<Integer> trainFilter = n -> true;
		IFilter<Integer> testFilter;
		if (tFilter != 1) { // 1 is for when no filter is used
			IFilter<Integer> temp = n -> n % tFilter != 1;	
			if (eFilter != 1) {
				testFilter = n -> ! temp.accept(n);
			}
			else { 
				testFilter = n -> true;
			}
			trainFilter = temp;
		}
		else { 
			trainFilter = n -> true;
			testFilter = n -> true;
		}
		
		// or like this?
		//		IFilter<Integer> trainFilter = tSplit == 1 ? n -> true : n -> n % (100-tSplit) != 1;
		//		IFilter<Integer> testFilter = eSplit == 1 ? n -> true : n -> (n % (eSplit != 1);
			
		// and we set the tag to know which filter was used
		modelDesc.put("tFilter", tFilter);
		modelDesc.put("eFilter", eFilter);

		// set up experiment
		EgBotExperiment experiment = trainExp(model, modelDesc, trainFilter, testFilter, files, tLabel);
		
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
		Log.i("Results at: "+Depot.getDefault().getLocalPath(experiment.getDesc()));
	}

	/**
	 * 
	 * @param model Do not use this again -- use the experiment.getModel() which may be a version loaded from the Depot cache
	 * @param modelDesc
	 * @param trainFilter
	 * @param testFilter
	 * @param files Training files
	 * @return
	 * @throws IOException
	 */
	public EgBotExperiment trainExp(IEgBotModel model, Desc<IEgBotModel> modelDesc, 
			IFilter<Integer> trainFilter,  IFilter<Integer> testFilter, List<File> files, String trainLabel) throws IOException 
	{
		// set up new experiment
		EgBotExperiment experiment = new EgBotExperiment();

		// set the model the experiment uses
		experiment.setModel(model, modelDesc);

		// Train
		// set the train filter		
		EgBotData trainData = new EgBotData(files, trainFilter);
		// set the train data the experiment uses
		Desc trainDataDesc = new Desc(trainLabel, EgBotData.class);
		trainDataDesc.put("use", "train");		
		experiment.setTrainData(trainData, trainDataDesc);
		
		// trained or just load pre-trained -- this can modify experiment's model ref
		EgBotDataLoader.train(experiment);		
		return experiment;
	}
	
}
