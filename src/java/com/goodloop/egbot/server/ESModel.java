package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.regex.Pattern;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.google.common.util.concurrent.ListenableFuture;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.depot.ModularXML;
import com.winterwell.es.client.ESHttpClient;
import com.winterwell.es.client.ESHttpResponse;
import com.winterwell.es.client.IndexRequestBuilder;
import com.winterwell.es.client.SearchRequestBuilder;
import com.winterwell.es.client.SearchResponse;
import com.winterwell.es.client.admin.CreateIndexRequest;
import com.winterwell.es.client.query.BoolQueryBuilder;
import com.winterwell.es.client.query.ESQueryBuilder;
import com.winterwell.es.client.query.ESQueryBuilders;
import com.winterwell.es.client.query.MoreLikeThisQueryBuilder;
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
import com.winterwell.utils.StrUtils;
import com.winterwell.utils.TodoException;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.containers.Pair2;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
//import static org.elasticsearch.common.xcontent.XContentFactory.*;
import com.winterwell.utils.web.SimpleJson;

public class ESModel implements IEgBotModel, IHasDesc, ModularXML {
	// location of Paulius' data set
	String filepath = System.getProperty("user.dir") + "/data/test_input/pauliusSample.json";
	
	final String indexName = "es-model";
	final String indexType = "qa";
	
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
		// take question, get most likely ans; 
//		bestTrainedQA = queryES with question	(call to generateMostLikely)
		String bestTrainedA = generateMostLikely(question, 0);
		
		// compare most likely ans to possible ans 
//		inputQA = question + possibleAnswer;
		Map inputQA = new ArrayMap<String, String>();
		inputQA.put("question", question);
		inputQA.put("answer", possibleAnswer);
		
//		index inputQA;
		ListenableFuture<ESHttpResponse> f = train1_doIt(inputQA);
		ESHttpResponse response;
		try {
			response = f.get();
		} catch (InterruptedException e) {
			throw Utils.runtime(e);
		} catch (ExecutionException e) {
			throw Utils.runtime(e);
		}
		response.check();
		Map<String, Object> r = response.getJsonMap(); // TODO was q-a known already? is so (which shouldnt happen for our experiments) dont delete on clean up
		
//		query ES with bestTrainedQA.answer and fixed ID (so it must return inputQA)
		double score = queryES(bestTrainedA, (String) inputQA.get("id"));
		
//		look at the score of the result
//		delete inputQA
		
		return score;
	}

	/**
	 * 
	 * @param bestTrainedA
	 * @return similarity between bestTrainedA and possibleAnswer (which has been stored as inputQA)
	 */
	private double queryES(String bestTrainedA, String inputQA_id) {
		Utils.check4null(bestTrainedA, inputQA_id);
		ESHttpClient esjc = new ESHttpClient();		
		SearchRequestBuilder s = esjc.prepareSearch(indexName);
		
		// build and run es query
		MoreLikeThisQueryBuilder sim = ESQueryBuilders.similar(bestTrainedA, Arrays.asList("answer"));
		ESQueryBuilder tq = ESQueryBuilders.termQuery("id", inputQA_id);
		BoolQueryBuilder q = ESQueryBuilders.boolQuery().must(tq).should(sim);
		s.setQuery(q);
		
		// get results
		SearchResponse sr = s.get();
		List<Map> hits = sr.getHits();
		throw new TodoException();
	}

	@Override
	public String sample(String question, int expectedAnswerLength) throws IOException {
		// use question as query and fetch most similar question from train
		return null;
	}

	/*
	 * get the answer of the question that's most similar to our input question, based on elastic's similarity scoring
	 */
	@Override
	public String generateMostLikely(String question, int expectedAnswerLength) throws IOException {
		List relatedQs = new RelatedQuestionFinder().run(question);
		List relatedAs = new ArrayList();
		
		for (int i=0; i<relatedQs.size(); i++) {
			if (relatedQs!=null &&  ! relatedQs.isEmpty()) {
				Object rq = relatedQs.get(i);
				double noOfAnswers = SimpleJson.get(rq, "answer_count");
				for (int j=0; j<noOfAnswers; j++) {
					Object answer = SimpleJson.get(rq, "answers", j);	
					boolean accepted = SimpleJson.get(rq, "answers", j, "is_accepted");
					if (accepted) {
						relatedAs.add(answer);
						break;
					}
				}
			}
		}	
		return (String) relatedAs.get(0);
	}

	@Override
	public void load() throws IOException {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void init(List<File> files, int num_epoch, String preprocessing, String wordEmbed) throws IOException {
		// set up our client
		ESHttpClient client = new ESHttpClient();
		// preparing for an index
		CreateIndexRequest preparedCreation = client.admin().indices().prepareCreate(indexName);
		preparedCreation.get().check();
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
	public String getModelConfig() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void train1(Map data) throws UnsupportedOperationException {
		train1_doIt(data);
	}

	private ListenableFuture<ESHttpResponse> train1_doIt(Map data) {
		// actually index data 
		ESHttpClient client = new ESHttpClient();
		String qa = (String)data.get("question") + (String)data.get("answer");
		assert ! data.containsKey("id") : data;
		String hashedQA = StrUtils.md5(qa);
		data.put("id", hashedQA);
		
		// create the index
		IndexRequestBuilder request = client.prepareIndex(indexName, indexType, hashedQA);
		
		// adding the data
		request.setBodyMap(data);
		return request.execute();
	}
	
}