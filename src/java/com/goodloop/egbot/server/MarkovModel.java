package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.eclipse.jetty.util.ajax.JSON;
import org.elasticsearch.index.analysis.WordDelimiterTokenFilterFactory;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.datalog.Rate;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
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
import com.winterwell.utils.Printer;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

public class MarkovModel implements IEgBotModel {
	
	ITrainable.Supervised<Cntxt, Tkn> wmc;
	/**
	 * we need this early for load()
	 */
	private final Desc desc;
	private String[] sig;
	private WordAndPunctuationTokeniser tokeniserFactory;
	private SitnStream ssFactory;
	public boolean loadSuccessFlag;

	public boolean isLoadSuccessFlag() {
		return loadSuccessFlag;
	}

	public MarkovModel() {
		sig = new String[] {"w-1", "w-2"};
		tokeniserFactory = new WordAndPunctuationTokeniser();
		ssFactory = new SitnStream(null, tokeniserFactory, sig);
		// NB: WWModel has its own desc which clashes with this and causes a bug :(
		// save load needs depot to be initialised
		Depot.getDefault().init();
		wmc = newModel();		
		desc = wmc instanceof IHasDesc? ((IHasDesc) wmc).getDesc() : new Desc<>("MSE-all", wmc.getClass());
		desc.setTag("egbot");
	}
	
	public void load() {
		// replace the newly made blank with a loaded copy if there is one
		Supervised<Cntxt, Tkn> _wmc = (ITrainable.Supervised<Cntxt, Tkn>) Depot.getDefault().get(desc);
		if (_wmc != null) {
			wmc = _wmc;
			loadSuccessFlag = true;
		}
	}
	
	void save() {
		// TODO: check with DW whether the code below makes sense
		// had to change it so that it uses the wmc's desc rather than the global desc variable because of the error below
		// java.lang.IllegalStateException: Desc mismatch: artifact-desc: Desc[w-1+w-2 null/WWModel/local/TPW=5000_sig=w-1, w-2_tr=1250, 2500, 12, 25/ce4f5336357e370c771aa5d4e1cfc709/w-1+w-2] != depot-desc: Desc[MSE-all egbot/WWModel/local/sig=w-1, w-2/MSE-all]
		// at com.winterwell.depot.Depot.safetySyncDescs(Depot.java:475)
		Desc d2 = ((IHasDesc) wmc).getDesc();
		assert d2.equals(desc);
		Depot.getDefault().put(desc, wmc);
	}
	/**
	 * TODO break this out into a common train-over-MSE-data class
	 * @throws IOException
	 */
	public void train () throws IOException {
		assert wmc != null;
		// load if we can
		load();
		// already done?
		if (loadSuccessFlag) {
			return;
		}
		// no -- train!
		assert wmc != null;
		
		// TODO refactor -- merge with ConstructEvaluationSet
		
		EgbotConfig config = new EgbotConfig();
		
		List<File> files;
		assert config.srcDataDir.isDirectory() : config.srcDataDir;
		if (false) {
			// zenodo data slimmed down to filter only q&a body_markdown using python script data-collection/slimming.py
			// Use this for extra speed if youve run the slimming script
			// python script data-collection/slimming.py
			files = Arrays.asList(new File(config.srcDataDir, "slim").listFiles());
		} else {
			File[] fs = config.srcDataDir.listFiles(new FilenameFilter() {				
				@Override
				public boolean accept(File dir, String name) {
					return name.startsWith("MathStackExchangeAPI_Part") 
							&& (name.endsWith(".json") || name.endsWith(".json.zip"));
				}
			});
			assert fs != null && fs.length > 0 : config.srcDataDir;
			files = Arrays.asList(fs);
		}
		// always have the same ordering
		Collections.sort(files);
		
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		for(File file : files) {
			System.out.println("File: "+file+"...");
	
			Gson gson = new Gson();
			// zip or plain json?
			Reader r;
			if (file.getName().endsWith(".zip")) {
				r = FileUtils.getZIPReader(file);
			} else {
				r = FileUtils.getReader(file);
			}
			JsonReader jr = new JsonReader(r);
			jr.beginArray();
						
			int c=0;
			while(jr.hasNext()) {
				Map qa = gson.fromJson(jr, Map.class);			
				Number answer_count = (Number) qa.get("answer_count");
				if (answer_count.intValue() > 0) {
					String body = SimpleJson.get(qa, "answers", 0, "body_markdown");
					SimpleDocument doc = new SimpleDocument(body);
					SitnStream ss2 = ssFactory.factory(doc);		
					
					for (Sitn<Tkn> sitn : ss2) {
						Cntxt prev = sitn.context;
						Tkn word = sitn.outcome;						
						wmc.train1(prev, word, 1);
					}
					c++;
					rate.plus(1);
					if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
				}			
			} 
			jr.close();
			
//			if (false) break;
		}
		// save trained model
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
		mm.train();
		mm.sample("what is a probability");
	}

	public  String sample(String q) {
		SitnStream ssq = ssFactory.factory(q);
		List<Sitn<Tkn>> list = Containers.getList(ssq);

		Sitn<Tkn> last = list.get(list.size()-1);
		
		// hack test by sampling
		Cntxt cntxt = last.context;
		int max_length = 30;
		String answer = "";
		for(int i=0; i<max_length; i++) {
			IFiniteDistribution<Tkn> marginal = (IFiniteDistribution<Tkn>) ((ICondDistribution<Tkn, Cntxt>)wmc).getMarginal(cntxt);
			// this is the most likely rather than a random sample
			Tkn sampled = marginal.getMostLikely();
			//Tkn sampled = ((ICond Distribution<Tkn, Cntxt>)wmc).sample(cntxt);
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
	 * @return
	 */
	public double scoreAnswer(String q, String t) {
		SitnStream ssq = ssFactory.factory(q);
		List<Sitn<Tkn>> qlist = Containers.getList(ssq);
		Sitn<Tkn> last = qlist.get(qlist.size()-1);
		Cntxt cntxt = last.context;
		
		SitnStream ssa = ssFactory.factory(t);		
		List<Sitn<Tkn>> alist = Containers.getList(ssq);
		
		double score = 0; 
		ICondDistribution<Tkn, Cntxt> cm = (ICondDistribution<Tkn, Cntxt>)wmc;
		// for each target word
		for (Sitn<Tkn> sitn : alist) {
			double p = cm.logProb(sitn.outcome, sitn.context); 
			//System.out.println(p);
			// add the log prob to the score
			score += p;
		}
		// avg the score and then return it
		return score/alist.size();
	}
}







