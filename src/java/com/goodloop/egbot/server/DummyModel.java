package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.depot.ModularXML;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
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
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;

// model that spits out pre-set answers given certain questions from Paulius' data set

public class DummyModel implements IEgBotModel, IHasDesc, ModularXML {
	// location of Paulius' data set
	String filepath = System.getProperty("user.dir") + "/data/test_input/pauliusSample.json";
	// question-answer pairs in the data set
	List qas = new ArrayList();
	// pattern for removing punctuation
	private final Pattern UNWANTED_SYMBOLS =
	        Pattern.compile("\\p{Punct}");
	
	public DummyModel() throws IOException {
		Log.i("File: "+filepath+"...");
		File file = new File(filepath);
		
		// read json
		Gson gson = new Gson(); 
		Reader r = FileUtils.getReader(file);
		JsonReader jr = new JsonReader(r);
		jr.beginArray();
		
		// add qa pair to global variable
		while(jr.hasNext()) {
			Map qa = gson.fromJson(jr, Map.class);
			qas.add(qa);
		} 
		jr.close();
	}

	public String getAnswer(String query) {
		String queryAnswer = "";
		
		// make lowercase and remove punct 
		query = query.toLowerCase();
		query = UNWANTED_SYMBOLS.matcher(query).replaceAll("");

		// go through every question in data set
		for (int i = 0; i < qas.size(); i++) {
			Map qa = (Map) qas.get(i);
			String pauliusQuestion = (String)qa.get("question");
			String pauliusAnswer = (String)qa.get("answer");
			
			// make lowercase and remove punct 
			pauliusQuestion = pauliusQuestion.toLowerCase();
			pauliusQuestion = UNWANTED_SYMBOLS.matcher(pauliusQuestion).replaceAll("");
			
			// compare query and existing question from data set
			if (query.equals(pauliusQuestion)) {
				queryAnswer = pauliusAnswer;
			}
		}
		return queryAnswer;
	}
	
	@Override
	public void train1(Map data) throws UnsupportedOperationException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void finishTraining() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean isReady() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void resetup() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Desc getDesc() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double scoreAnswer(String question, String possibleAnswer) throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String sample(String question, int expectedAnswerLength) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String generateMostLikely(String question, int expectedAnswerLength) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void load() throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setTrainSuccessFlag(boolean b) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setLoadSuccessFlag(boolean b) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void init(List<File> files, int num_epoch) throws IOException {
		// TODO Auto-generated method stub
		
	}
}







