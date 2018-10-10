package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.eclipse.jetty.util.ajax.JSON;
import org.elasticsearch.index.analysis.WordDelimiterTokenFilterFactory;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.maths.stats.distributions.cond.Sitn;
import com.winterwell.maths.stats.distributions.cond.WordMarkovChain;
import com.winterwell.nlp.corpus.SimpleDocument;
import com.winterwell.nlp.io.SitnStream;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.nlp.io.WordAndPunctuationTokeniser;
import com.winterwell.utils.Printer;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.web.SimpleJson;

public class MarkovModel {
	
	WordMarkovChain<Tkn> wmc;
	private Desc<WordMarkovChain> desc;
	private String[] sig;

	public MarkovModel() {
		sig = new String[] {"w-1", "w-2"};
		desc = new Desc<>("MSE-all",WordMarkovChain.class)
				.setTag("egbot");
		desc.put("sig", Printer.toString(sig));
	}
	
	void load() {
		wmc = Depot.getDefault().get(desc);
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
		wmc = new WordMarkovChain<>();
		for (int fileNumber=1; fileNumber<=8; fileNumber++) {
			File file = new File("data/build/MathStackExchangeAPI_Part_" + fileNumber + ".json");
	
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
					if (c % 1000 == 0) System.out.println(c+"...");
				}			
			} 
		}
		save();
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
			Tkn sampled = wmc.sample(cntxt);
			System.out.println(sampled);
			cntxt = new Cntxt(sig, sampled, cntxt.getBits()[0]);
		}
	}

}
