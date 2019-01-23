package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.depot.ModularXML;
import com.winterwell.maths.ITrainable;
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
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.containers.Pair2;
import com.winterwell.utils.log.Log;

public class MarkovModel implements IEgBotModel, IHasDesc, ModularXML {
	
//	ITrainable.Supervised<Cntxt, Tkn> 
	WWModel<Tkn> wmc;
	/**
	 * we need this early for load()
	 * 
	 * This is a desc of the guts ie wmc
	 */
	public Desc wmcDesc;
	public Desc<IEgBotModel> mmDesc;
	private String[] sig;
	private WordAndPunctuationTokeniser tokeniserFactory;
	private SitnStream ssFactory;
	
	public WordAndPunctuationTokeniser getTokeniserFactory() {
		return tokeniserFactory;
	}
	
	// true when model was successfully loaded
	public boolean loadSuccessFlag;
		
	// true once training finished on all egbot files
	public boolean trainSuccessFlag;
	private final String tag = "egbot";

	public MarkovModel() {
		loadSuccessFlag = false;
		trainSuccessFlag = false;
		sig = new String[] {"w-1", "w-2"};
		tokeniserFactory = new WordAndPunctuationTokeniser();
		ssFactory = new SitnStream(null, tokeniserFactory, sig);
		
		// save load needs depot to be initialised
		wmc = (WWModel<Tkn>) newModel(); //guts
		Depot.getDefault().init();
		wmcDesc = wmc.getDesc();  
		wmcDesc.setTag(tag);
		
		mmDesc = new Desc(wmcDesc.getName(), MarkovModel.class);
		mmDesc.setTag(tag);
		mmDesc.addDependency("guts", wmcDesc);  
		
		// ??how should we say these modules are only for this object,
		// without creating a dependency circular reference mess??
		// TODO add a feature to Depot or Desc to say "don't use modular saving please"
		String p = "MarkovModel:w-1,w-2"; // Hack - use a fixed string. NB: we tried toString(), but that has unstable outputs e.g. "MarkovModel@591f989e";
		for(IHasDesc md : getModules()) {
			md.getDesc().put("parent", p);
		}
	}

	/**
	 * The load/save is done by {@link EgBotDataLoader} using Depot on the whole MarkovModel object.
	 * 
	 * Internal state is not separately load/saved
	 */
	public void load() {
//		Log.d("load MarkovModel guts from "+wmcDesc+"...");
//		// replace the newly made blank with a loaded copy if there is one
//		Supervised<Cntxt, Tkn> _wmc = (ITrainable.Supervised<Cntxt, Tkn>) Depot.getDefault().get(wmcDesc);
//		
//		if (_wmc != null) {
//			wmc = _wmc;
//			loadSuccessFlag = true;
//			Log.d("Loaded succesfully :)");
//		}
//		else {
//			Log.d("Sorry, couldn't load ");
//		}
	}
	
	public void save() {
//		// NB: had to change it so that it uses the wmc's desc rather than the global desc variable because of the error below
//		// java.lang.IllegalStateException: Desc mismatch: artifact-desc: Desc[w-1+w-2 null/WWModel/local/TPW=5000_sig=w-1, w-2_tr=1250, 2500, 12, 25/ce4f5336357e370c771aa5d4e1cfc709/w-1+w-2] != depot-desc: Desc[MSE-all egbot/WWModel/local/sig=w-1, w-2/MSE-all]
//		// at com.winterwell.depot.Depot.safetySyncDescs(Depot.java:475)
//		Desc d2 = ((IHasDesc) wmc).getDesc(); 
//		assert d2.equals(wmcDesc);
//		Depot.getDefault().put(wmcDesc, wmc);
//		
//		// saving model TODO: is this necessary?
//		Depot.getDefault().put(mmDesc, this);
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
		trainSuccessFlag = true; // TODO: set it at the end
		// save model
		save();
	}
		
	private ITrainable.Supervised<Cntxt, Tkn> newModel() {
		if (false) {
			// a simple markov model -- will eat memory!Model
			return new WordMarkovChain<>();
		}
		WWModel<Tkn> model = new WWModelFactory().fullFromSig(Arrays.asList(sig));
		return model;
	}

	public static void main(String[] args) throws IOException {
		MarkovModel mm = new MarkovModel();		
		Desc<IEgBotModel> trainedModelDesc = mm.getDesc();
		MarkovModel trainedMarkov = (MarkovModel) Depot.getDefault().get(trainedModelDesc);
		
		// set up experiment
		EgBotExperiment e1 = new EgBotExperiment();
		// set the model the experiment uses
		e1.setModel(mm, trainedModelDesc);
		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 100 != 1;
		IFilter<Integer> testFilter = n -> ! trainFilter.accept(n);
		// decide on train data version (e.g. MSE-full, MSE-20, paul-20)
		String trainLabel = "MSE-full";
		// load the list of files
		List<File> files = EgBotDataLoader.setup(trainLabel);

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
			cntxt = new Cntxt(sig, sampled, cntxt.getBits()[0]); //TODO: currently this is written up to generate only bigrams, fix this so that depending on the sig length n, it generates ngrams
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
			score += p;
			count++;
		}
		// avg the score
		return score/count;
	}
	
	/**
	 * score model's ability to guess the correct answer from a set of answers
	 * @param q question 
	 * @param t target
	 * @param a answers
	 * @return 1 if correct, 0 if incorrect
	 * @throws IOException
	 */
	public int scorePickBest(String q, String t, ArrayList<String> a) throws IOException {
		int bestAnsIdx = pickBest(q, t, a);
		if (a.indexOf(t) == bestAnsIdx) return 1; 
		else return 0;
	}
	
	/**
	 * get model's best guess from a selection of answers, of which one is correct
	 * @param q question 
	 * @param t target
	 * @param a answers
	 * @return index of answer deemed to be the best guess
	 * @throws IOException 
	 */
	public int pickBest(String q, String t, ArrayList<String> a) throws IOException {
		double bestAvg = -999; // artificially low score (TODO: is this correct way of doing it?)
		int bestAnsIdx = -1;

		// for each answer in the list
		for (String ans : a) {
			double wordScores = 0;
			int count = 0;
			String body = q + " " + ans; // concatenate q and a
			SimpleDocument doc = new SimpleDocument(body);
			SitnStream ss2 = ssFactory.factory(doc);
			for (Sitn<Tkn> sitn : ss2) {					
				// add the log prob to the score
				double p = ((ACondDistribution<Tkn, Cntxt>) wmc).logProb(sitn.outcome, sitn.context); // sometimes returns -Infinity
				wordScores += p; // add the log prob to the score (so as to calculate the answer avg later)
				count++;
			}
			// avg the word scores
			double avg = wordScores/count;
			// save the index of the answer that has the best average word score
			if (avg > bestAvg) {
				bestAnsIdx = a.indexOf(ans);
				bestAvg = avg;
			}
		}
		return bestAnsIdx;
	}
	
	/**
	 * initialise any model parameters to prepare for training
	 */
	public void init(List<File> files) throws IOException {
	}
	
	@Override
	public boolean isReady() {
		return loadSuccessFlag || trainSuccessFlag;
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
					(IHasDesc) wmc,
					tokeniserFactory
			};
		}
		return new IHasDesc[] {tokeniserFactory};
	}
	
	public boolean isLoadSuccessFlag() {
		return loadSuccessFlag;
	}

	public void setLoadSuccessFlag(boolean loadSuccessFlag) {
		this.loadSuccessFlag = loadSuccessFlag;
	}
	
	public void setTrainSuccessFlag(boolean trainSuccessFlag) {
		this.trainSuccessFlag = trainSuccessFlag;
	}
	
	public Desc<IEgBotModel> getDesc() {
		return mmDesc;
	}
}







