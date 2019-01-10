package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.eclipse.jetty.util.ajax.JSON;
import org.elasticsearch.index.analysis.WordDelimiterTokenFilterFactory;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.datalog.Rate;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.depot.ModularXML;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.ITrainable;
import com.winterwell.maths.ITrainable.Supervised;
import com.winterwell.maths.stats.distributions.IDistributionBase;
import com.winterwell.maths.stats.distributions.cond.ACondDistribution;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.maths.stats.distributions.cond.ICondDistribution;
import com.winterwell.maths.stats.distributions.cond.Sitn;
import com.winterwell.maths.stats.distributions.cond.WWModel;
import com.winterwell.maths.stats.distributions.cond.WWModelFactory;
import com.winterwell.maths.stats.distributions.cond.WordMarkovChain;
import com.winterwell.maths.stats.distributions.discrete.IFiniteDistribution;
import com.winterwell.nlp.corpus.SimpleDocument;
import com.winterwell.nlp.io.SitnStream;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.nlp.io.WordAndPunctuationTokeniser;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.Printer;
import com.winterwell.utils.TodoException;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

public class MarkovModel implements IEgBotModel, IHasDesc, ModularXML {
	
	ITrainable.Supervised<Cntxt, Tkn> wmc;
	/**
	 * we need this early for load()
	 * 
	 * This is a desc of the guts ie wmc
	 */
	public final Desc desc;
	private String[] sig;
	private WordAndPunctuationTokeniser tokeniserFactory;
	private SitnStream ssFactory;
	
	// true when model was successfully loaded
	public boolean loadSuccessFlag;
		
	// true once training finished on all egbot files
	public boolean trainSuccessFlag;

	public MarkovModel() {
		loadSuccessFlag = false;
		trainSuccessFlag = false;
		sig = new String[] {"w-1", "w-2"};
		tokeniserFactory = new WordAndPunctuationTokeniser();
		ssFactory = new SitnStream(null, tokeniserFactory, sig);
		// NB: WWModel has its own desc which clashes with this and causes a bug :(
		// save load needs depot to be initialised
		Depot.getDefault().init();
		wmc = newModel();		
		desc = wmc instanceof IHasDesc? ((IHasDesc) wmc).getDesc() : new Desc<>("MSE-mm", wmc.getClass()); 
		desc.setTag("egbot");
	}

	public void load() {
		Log.d("load MarkovModel guts from "+desc+"...");
		// replace the newly made blank with a loaded copy if there is one
		Supervised<Cntxt, Tkn> _wmc = (ITrainable.Supervised<Cntxt, Tkn>) Depot.getDefault().get(desc);
		if (_wmc != null) {
			wmc = _wmc;
			loadSuccessFlag = true;
			Log.d("Loaded succesfully :)");
		}
		else {
			Log.d("Sorry, couldn't load ");
		}
	}
	
	public void save() {
		// NB: had to change it so that it uses the wmc's desc rather than the global desc variable because of the error below
		// java.lang.IllegalStateException: Desc mismatch: artifact-desc: Desc[w-1+w-2 null/WWModel/local/TPW=5000_sig=w-1, w-2_tr=1250, 2500, 12, 25/ce4f5336357e370c771aa5d4e1cfc709/w-1+w-2] != depot-desc: Desc[MSE-all egbot/WWModel/local/sig=w-1, w-2/MSE-all]
		// at com.winterwell.depot.Depot.safetySyncDescs(Depot.java:475)
		Desc d2 = ((IHasDesc) wmc).getDesc();
		assert d2.equals(desc);
		Depot.getDefault().put(desc, wmc);
	}
	
	@Override
	public void train1(Map qa) throws UnsupportedOperationException {
		String question_body = (String) qa.get("question");
		String answer_body = (String) qa.get("answer");
		String body = question_body + " " + answer_body;
		SimpleDocument doc = new SimpleDocument(body);
		SitnStream ss2 = ssFactory.factory(doc);	
		for (Sitn<Tkn> sitn : ss2) {
			Cntxt prev = sitn.context;
			Tkn word = sitn.outcome;						
			wmc.train1(prev, word, 1);				
		}
		// save model
		save();
	}
		
	private ITrainable.Supervised<Cntxt, Tkn> newModel() {
		if (false) {
			// a simple markov model -- will eat memory!
			return new WordMarkovChain<>();
		}
		WWModel<Tkn> model = new WWModelFactory().fullFromSig(Arrays.asList(sig));
		return model;
	}

	public static void main(String[] args) throws IOException {
		MarkovModel mm = new MarkovModel();		
		Desc<IEgBotModel> trainedModelDesc = mm.getDesc();
		IEgBotModel trainedMarkov = Depot.getDefault().get(trainedModelDesc);
		
		// set up experiment
		EgBotExperiment e1 = new EgBotExperiment();
		// set the model the experiment uses
		e1.setModel(mm, trainedModelDesc);
		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 100 != 1;
		IFilter<Integer> testFilter = n -> ! trainFilter.accept(n);
		// load the list of egbot files
		List<File> files = EgBotDataLoader.setup();

		// Train
		// set the train filter		
		EgBotData trainData = new EgBotData(files, trainFilter);
		// set the train data the experiment uses
		Desc<EgBotData> trainDataDesc = new Desc("MSE-data", EgBotData.class);
		e1.setTrainData(trainData, trainDataDesc);
		if (trainedMarkov==null) {
			// do training
			EgBotDataLoader.train(e1);
		}
		mm.sample("what is a probability", 30);
	}
	
	/**
	 * Get a sample of words from the model.
	 * 
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	public  String sample(String q, int expectedAnswerLength) throws IOException {
		SitnStream ssq = ssFactory.factory(q);
		List<Sitn<Tkn>> list = Containers.getList(ssq);

		Sitn<Tkn> last = list.get(list.size()-1);
		
		// hack test by sampling
		Cntxt cntxt = last.context;
		String answer = "";
		for(int i=0; i<expectedAnswerLength; i++) {
			IFiniteDistribution<Tkn> marginal = (IFiniteDistribution<Tkn>) ((ICondDistribution<Tkn, Cntxt>)wmc).getMarginal(cntxt);
			Tkn sampled = marginal.sample();
			if (Tkn.END_TOKEN.equals(sampled)) {
				break;
			}			
			answer = answer + " " + sampled.toString();
			cntxt = new Cntxt(sig, sampled, cntxt.getBits()[0]);
		}
		return answer;
	}
	
	/**
	 * Get the most likely series of words from the model.
	 *  
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	public String generateMostLikely(String q, int expectedAnswerLength) throws IOException {
		SitnStream ssq = ssFactory.factory(q);
		List<Sitn<Tkn>> list = Containers.getList(ssq);

		Sitn<Tkn> last = list.get(list.size()-1);
		
		Cntxt cntxt = last.context;
		String answer = "";
		for(int i=0; i<expectedAnswerLength; i++) {
			IFiniteDistribution<Tkn> marginal = (IFiniteDistribution<Tkn>) ((ICondDistribution<Tkn, Cntxt>)wmc).getMarginal(cntxt);
			// this is the most likely rather than a random sample
			Tkn sampled = marginal.getMostLikely();
			if (Tkn.END_TOKEN.equals(sampled)) {
				break;
			}			
			answer = answer + " " + sampled.toString();
			cntxt = new Cntxt(sig, sampled, cntxt.getBits()[0]);
		}
		return answer;
	}
	
	/**
	 * score answer for given question (where the score is the avg log probability of each word in the answer being predicted)
	 * @param q question
	 * @param t target answer
	 * @return 
	 */
	public double scoreAnswer(String q, String t) {
		double score = 0; 
		String body = q + " " + t;
		SimpleDocument doc = new SimpleDocument(body);
		SitnStream ss2 = ssFactory.factory(doc);
		int count = 0;
		for (Sitn<Tkn> sitn : ss2) {					
			// add the log prob to the score
			double p = ((ACondDistribution<Tkn, Cntxt>) wmc).logProb(sitn.outcome, sitn.context); 
			//System.out.println("Example log prob: " + p);
			score += p;
			count++;
		}
		// avg the score
		return score/count;
	}
	
	/**
	 * score best guess 
	 * @param q question 
	 * @param t target
	 * @param a answers
	 * @return index of answer deemed to be the best guess
	 */
	public int scorePickBest(String q, String t, ArrayList<String> a) {
		double score = 0; 
		double bestAvg = -999; // artifically low score
		int bestAnsIdx = -1;
		
		for (String ans : a) {
			String body = q + " " + ans;
			SimpleDocument doc = new SimpleDocument(body);
			SitnStream ss2 = ssFactory.factory(doc);
			int count = 0;
			for (Sitn<Tkn> sitn : ss2) {					
				// add the log prob to the score
				double p = ((ACondDistribution<Tkn, Cntxt>) wmc).logProb(sitn.outcome, sitn.context); 
				//System.out.println("Example log prob: " + p);
				score += p;
				count++;
			}
			// avg the score
			double avg = score/count;
			if (bestAvg > avg) {
				bestAnsIdx = a.indexOf(ans);
			}
		}
		if (a.indexOf(t) == bestAnsIdx) return 1; 
		else return 0;
	}
	
	/**
	 * initialise any model parameters to prepare for training
	 */
	public void init(List<File> files) throws IOException {
	}
	
	public Desc getDesc() {
		Desc mmDesc = new Desc(desc.getName(), MarkovModel.class);
		mmDesc.addDependency("guts", desc);
		return mmDesc;
	}

	public ITrainable.Supervised<Cntxt, Tkn> getWmc() {
		return wmc;
	}

	@Override
	public boolean isReady() {
		return loadSuccessFlag;
	}

	@Override
	public void finishTraining() {
	}

	@Override
	public void resetup() {
	}

	@Override
	public IHasDesc[] getModules() {
		if (wmc instanceof IHasDesc) {
			return new IHasDesc[] {
					(IHasDesc) wmc
			};
		}
		return null;
	}
	
	public boolean isLoadSuccessFlag() {
		return loadSuccessFlag;
	}

	public void setLoadSuccessFlag(boolean loadSuccessFlag) {
		this.loadSuccessFlag = loadSuccessFlag;
	}
	
	public boolean isTrainSuccessFlag() {
		return trainSuccessFlag;
	}
	
	public void setTrainSuccessFlag(boolean trainSuccessFlag) {
		this.trainSuccessFlag = trainSuccessFlag;
	}
}







