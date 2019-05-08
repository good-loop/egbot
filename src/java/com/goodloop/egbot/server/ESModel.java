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

import com.goodloop.egbot.EgbotConfig;
import com.google.common.util.concurrent.ListenableFuture;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.depot.ModularXML;
import com.winterwell.es.IESRouter;
import com.winterwell.es.StdESRouter;
import com.winterwell.es.client.DeleteRequestBuilder;
import com.winterwell.es.client.ESConfig;
import com.winterwell.es.client.ESHttpClient;
import com.winterwell.es.client.ESHttpResponse;
import com.winterwell.es.client.IndexRequestBuilder;
import com.winterwell.es.client.KRefresh;
import com.winterwell.es.client.SearchRequestBuilder;
import com.winterwell.es.client.SearchResponse;
import com.winterwell.es.client.admin.CreateIndexRequest;
import com.winterwell.es.client.admin.DeleteIndexRequest;
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
import com.winterwell.utils.Dep;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.StrUtils;
import com.winterwell.utils.TodoException;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.containers.Pair2;
import com.winterwell.utils.io.ConfigFactory;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
//import static org.elasticsearch.common.xcontent.XContentFactory.*;
import com.winterwell.utils.web.SimpleJson;
import com.winterwell.web.app.AMain;
import com.winterwell.web.app.AppUtils;

public class ESModel implements IEgBotModel, IHasDesc, ModularXML {
	// location of Paulius' data set
	String filepath = System.getProperty("user.dir") + "/data/test_input/pauliusSample.json";
	
	final static String indexName = "egbot.esquestion";
	final static String indexType = "qa";
	
	private final Desc<IEgBotModel> desc = new Desc<IEgBotModel>("ESModel", ESModel.class).setTag("egbot").setVersion(0);
	boolean trainSuccessFlag;
	
	/**
	 * get ES similarity score for the most likely answer and known possible answer 
	 * @param question, possibleAnswer
	 * @returns similarity score
	 */
	@Override
	public double scoreAnswer(String question, String possibleAnswer) throws IOException {
		// take question, get most likely ans; 
		String bestTrainedA = generateMostLikely(question, 0);
		
		// compare most likely ans to possible ans 
		Map inputQA = new ArrayMap<String, String>();
		inputQA.put("question", question);
		inputQA.put("answer", possibleAnswer);
		
		//index inputQA and return inputQA with added field id (which is the hashed q+a)
		inputQA = train1_doIt(inputQA, true);
				
		//query ES with bestTrainedQA.answer and fixed ID (so it must return inputQA)
		double score = queryES(bestTrainedA, (String) inputQA.get("id"));
		
		//look at the score of the result
		System.out.println("Score: " + score);
		
		//delete inputQA
		ESHttpClient client = new ESHttpClient();
		DeleteRequestBuilder preparedDeletion = client.prepareDelete(indexName, indexType, (String)inputQA.get("id"));
		preparedDeletion.get().check();
		
		return score;
	}

	/**
	 * ask ES for the similarity score
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
		if(!hits.isEmpty()) {
			Map hit = hits.get(0);
			double score = (double) hit.get("_score");
			return score;
		}			
		else {
			Log.d("Couldn't get results for this query: " + q.toString()); 
			throw new IndexOutOfBoundsException(); 
		}
	}

	/**
	 * get the answer of the question that's most similar to our input question, based on elastic's similarity scoring
 	 * @param question, expectedAnswerLength
 	 * @returns most similar answer
	 */
	@Override
	public String generateMostLikely(String question, int expectedAnswerLength) throws IOException {
		List relatedQAs = new RelatedESquestion().run(question);
		List relatedAs = new ArrayList();
		
		if (relatedQAs!=null &&  ! relatedQAs.isEmpty()) {
			int qaSize = relatedQAs.size();
			for (int i=0; i<qaSize ; i++) {
				Object rqa = relatedQAs.get(i);
				Object answer = SimpleJson.get(rqa, "answer");
				//Object answer = SimpleJson.get(rq, "answers", j);	
				//boolean accepted = SimpleJson.get(rq, "answers", j, "is_accepted");
				relatedAs.add(answer);
				break;
			}
		}	
		return (String) relatedAs.get(0);
	}

	/**
	 * initialise the model by setting up ES and preparing for an index
	 */
	@Override
	public void init(List<File> files, int num_epoch, String preprocessing, String wordEmbed) throws IOException {
		// set up es config
		ESConfig config = AppUtils.getConfig("egbot", ESConfig.class, null);
		ESConfig esc = ConfigFactory.get().getConfig(ESConfig.class);

		// set up our client
		ESHttpClient client = new ESHttpClient(esc);
		Dep.setIfAbsent(ESHttpClient.class, client);
		assert config != null;
		// Is the config the IESRouter?
		if (config instanceof IESRouter) {
			Dep.setIfAbsent(IESRouter.class, (IESRouter) config);
		} else {
			// nope - use a default
			Dep.setIfAbsent(IESRouter.class, new StdESRouter());
		}

		// preparing for an index
		boolean indexExists = client.admin().indices().indexExists(indexName);
		if (!indexExists) {
			CreateIndexRequest preparedCreation = client.admin().indices().prepareCreate(indexName);
			preparedCreation.get().check();
		}		
	}

	/**
	 * index the training data 
	 * @param data
	 */
	@Override
	public void train1(Map data) throws UnsupportedOperationException {
		train1_doIt(data, false);		
	}

	/**
	 * index the training data, with an option to say whether ES should do a forced re-index (if true, it ensures document changes appear in search results immediately)
	 * @param data, forceReindex
	 * @return q+a data with added field id (useful for ES indexing and search)
	 */
	private Map train1_doIt(Map data, boolean forceReindex) {
		// actually index data 
		ESHttpClient client = new ESHttpClient();
		String qa = (String)data.get("question") + " " + (String)data.get("answer");
		assert ! data.containsKey("id") : data;
		String hashedQA = StrUtils.md5(qa);
		data.put("id", hashedQA);
		
		// create the index
		IndexRequestBuilder request = client.prepareIndex(indexName, indexType, hashedQA);
		
		// adding the data
		request.setBodyMap(data);
		if (forceReindex) 
			request.setRefresh(KRefresh.TRUE);
		
		// get the response
		ListenableFuture<ESHttpResponse> f = request.execute();
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
		
		return data;
	}
	
	/**
	 * random sample (not useful here)
	 */
	@Override
	public String sample(String question, int expectedAnswerLength) throws IOException {
		return null;
	}
	
	@Override
	public void finishTraining() {
		trainSuccessFlag = true;		
	}

	@Override
	public boolean isReady() {
		return trainSuccessFlag;
	}

	@Override
	public void resetup() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Desc getDesc() {
		assert desc != null;
		return desc;
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
		return "";
	}
	
	@Override
	public void load() throws IOException {
		// TODO Auto-generated method stub
	}	
}