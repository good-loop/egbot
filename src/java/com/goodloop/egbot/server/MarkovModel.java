package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
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
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.ITrainable;
import com.winterwell.maths.ITrainable.Supervised;
import com.winterwell.maths.stats.distributions.cond.ACondDistribution;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.maths.stats.distributions.cond.ICondDistribution;
import com.winterwell.maths.stats.distributions.cond.Sitn;
import com.winterwell.maths.stats.distributions.cond.WWModel;
import com.winterwell.maths.stats.distributions.cond.WWModelFactory;
import com.winterwell.maths.stats.distributions.cond.WordMarkovChain;
import com.winterwell.nlp.corpus.SimpleDocument;
import com.winterwell.nlp.io.SitnStream;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.nlp.io.WordAndPunctuationTokeniser;
import com.winterwell.utils.Printer;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

public class MarkovModel {
	
	ITrainable.Supervised<Cntxt, Tkn> wmc;
	private Desc desc;
	private String[] sig;

	public MarkovModel() {
		sig = new String[] {"w-1", "w-2"};
		desc = new Desc<>("MSE-all", newModel().getClass())
				.setTag("egbot");
		desc.put("sig", Printer.toString(sig));
	}
	
	void load() {
		wmc = (ITrainable.Supervised<Cntxt, Tkn>) Depot.getDefault().get(desc);
	}
	
	void save() {
		Depot.getDefault().put(desc, wmc);
	}
	
	public void train () throws IOException {
		// already done?
		load();
		if (wmc!=null) {
			return;
		}
		wmc = newModel();
		
		EgbotConfig config = new EgbotConfig();
		
		List<File> files;
		if (false) {
			// zenodo data slimmed down to filter only q&a body_markdown using python script data-collection/slimming.py
			// Use this for extra speed if youve run the slimming script
			// python script data-collection/slimming.py
			files = Arrays.asList(new File(config.srcDataDir, "slim").listFiles());
		} else {
			files = Arrays.asList(config.srcDataDir.listFiles(new FilenameFilter() {				
				@Override
				public boolean accept(File dir, String name) {
					return name.startsWith("MathStackExchangeAPI_Part") && name.endsWith(".json");
				}
			}));
		}
		// always have the same ordering
		Collections.sort(files);
		
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		for(File file : files) {
			System.out.println("File: "+file+"...");
//		for (int fileNumber=1; fileNumber<=8; fileNumber++) {
//			// zenodo data slimmed down to filter only q&a body_markdown using python script data-collection/slimming.py
//			File file = new File("data/build/slim/MathStackExchangeAPI_Part_" + fileNumber + ".json");
	
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
			
			WordAndPunctuationTokeniser tokeniser = new WordAndPunctuationTokeniser();
			SitnStream ss = new SitnStream(null, tokeniser, sig);
			int c=0;
			while(jr.hasNext()) {
				Map qa = gson.fromJson(jr, Map.class);			
				Number answer_count = (Number) qa.get("answer_count");
				if (answer_count.intValue() > 0) {
					String body = SimpleJson.get(qa, "answers", 0, "body_markdown");
					SimpleDocument doc = new SimpleDocument(body);
					SitnStream ss2 = ss.factory(doc);		
					
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
		}
		save();
	}
	
	private ITrainable.Supervised<Cntxt, Tkn> newModel() {
		if (true) {
			// a simple markov model -- will eat memory!
			return new WordMarkovChain<>();
		}
		WWModel<Tkn> model = new WWModelFactory().fullFromSig(Arrays.asList(sig));
		return model;
	}

	public static void main(String[] args) throws IOException {
		MarkovModel mm = new MarkovModel();
		mm.train();
		mm.sample(10);
	}

	private void sample(int n) {
		// hack test by sampling
		Cntxt cntxt = new Cntxt(sig, Tkn.START_TOKEN, Tkn.START_TOKEN);
		for(int i=0; i<n; i++) {
			Tkn sampled = ((ICondDistribution<Tkn, Cntxt>)wmc).sample(cntxt);
			System.out.println(sampled);
			cntxt = new Cntxt(sig, sampled, cntxt.getBits()[0]);
		}
	}

}
