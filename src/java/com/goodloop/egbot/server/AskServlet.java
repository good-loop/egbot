package com.goodloop.egbot.server;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.utils.Proc;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.threads.Actor;
import com.winterwell.utils.web.SimpleJson;
import com.winterwell.web.ajax.JSend;
import com.winterwell.web.app.IServlet;
import com.winterwell.web.app.WebRequest;

import jep.Jep;
import jep.JepConfig;
import jep.JepException;

public class AskServlet implements IServlet {
	
	String failAnswer = "Hmm I couldn't figure it out ...";

	@Override
	public void process(WebRequest state) throws Exception {
		String q = state.get("q");

		Object answer;
		List relatedQs = findRelatedQuestion(q);
		List relatedAs = findRelatedAnswer(relatedQs);
		
		//TODO: fix trained mm loading, the problem seems to be that it generates a different dependency signature every time (as can be seen in the console); but it does this only here, not in training (how strange)
		MarkovModel mm = new MarkovModel(); 
		System.out.println("Model desc: " + mm.getDesc());
		Object generatedAnswer = generateAnswer(mm, q, "MSE-20", 100, 1);
		//Object generatedAnswer = generateAnswer(new LSTM(), q, "MSE-20", 100, 1); 

		ArrayMap data = new ArrayMap(
			"relatedQs", relatedQs,
			"relatedAs", relatedAs,
			"generatedAnswer", generatedAnswer
			);
		JSend jsend = new JSend(data);
		jsend.send(state);
	}
	
	/**
	 * run the python LSTM test script using the command line
	 */
	private Object generateAnswerCL(String q) throws Exception {		
		File inputTextFile = File.createTempFile("q", ".txt");
		FileUtils.write(inputTextFile, q);
		// TODO have lstmTestModel.py read from the file parameter
        Proc proc = new Proc("python lstmTestModel.py "+inputTextFile.getAbsolutePath());
        proc.start();
        proc.waitFor();
        String answer = proc.getOutput();
        proc.close();
		return answer;
	}
	
	/**
	 * use trained LSTM model to generate an answer
	 */
	private Object generateAnswer(IEgBotModel model, String q, String trainLabel, int tFilter, int eFilter) throws Exception {
		System.out.println("Loading model ...");
		Desc<IEgBotModel> modelDesc = model.getDesc();
		modelDesc.put("train", trainLabel);
		modelDesc.put("tFilter", tFilter);
		modelDesc.put("eFilter", eFilter);
		IEgBotModel pretrained = Depot.getDefault().get(modelDesc);
		// do we have a trained model that fits the description?
		if (pretrained!=null) {
			// replace the untrained with the trained
			Log.d("Using pre-trained model");
			model = pretrained;
			model.setLoadSuccessFlag(true);
			// generate answer
			System.out.println("Generating answer ...");
			String answer = model.generateMostLikely(q, 30);	
			System.out.println(answer);
			return answer;
		}		
		Log.d("Error: Couldn't find trained model");
		return failAnswer;
	}
	
	/**
	 * search ES and filter for questions that have accepted answers
	 * @param q 
	 * @return answered questions
	 */
	private List findRelatedQuestion(String q) {
		// By Analogy to previous Q
		// Find related Qs
		List relatedQsES = new RelatedQuestionFinder().run(q);
		List relatedQs = new ArrayList();
		
		// possibly redundant code
		for (int i=0; i<relatedQsES.size(); i++) {
			if (relatedQsES!=null &&  ! relatedQsES.isEmpty()) {
				Object rq = relatedQsES.get(i);
				double noOfAnswers = SimpleJson.get(rq, "answer_count");
				for (int j=0; j<noOfAnswers; j++) {
					Object answer = SimpleJson.get(rq, "answers", j);	
					boolean accepted = SimpleJson.get(rq, "answers", j, "is_accepted");
					if (accepted) {
						relatedQs.add(rq);
					}
				}
			}
		} 
		
		return relatedQs;
	}

	/**
	 * filter for answers that were accepted
	 * @param q 
	 * @return list of answers
	 */
	private  List findRelatedAnswer(List relatedQs) {
		// TODO adapt related Q into an answer
		List relatedAs = new ArrayList();
		
		// possibly redundant code
		for (int i=0; i<relatedQs.size(); i++) {
			if (relatedQs!=null &&  ! relatedQs.isEmpty()) {
				Object rq = relatedQs.get(i);
				double noOfAnswers = SimpleJson.get(rq, "answer_count");
				for (int j=0; j<noOfAnswers; j++) {
					Object answer = SimpleJson.get(rq, "answers", j);	
					boolean accepted = SimpleJson.get(rq, "answers", j, "is_accepted");
					if (accepted) {
						relatedAs.add(answer);
					}
				}
			}
		}
		return relatedAs;
	}
	
	static JEPActor jepActor = new JEPActor();
	
	private synchronized Object generateAnswerJEP(String q) throws Exception {
		JEPCall msg = new JEPCall("generateResults('" + q + "')");
		jepActor.send(msg);
		while(msg.output==null) {
			Utils.sleep(10);
		}
		return msg.output;
	}
	
	private Object generateAnswerJavaTF(String q) throws Exception {
		try (Graph g = new Graph()) {
		      final String value = "Hello from " + TensorFlow.version();
		      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
		        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
		      }
		      try (Session s = new Session(g);
		          Tensor output = s.runner().fetch("MyConst").run().get(0)) {
		        System.out.println(new String(output.bytesValue(), "UTF-8"));
		      }
		}
		return "";
	}
}


class JEPActor extends Actor<JEPCall> {

	@Override
	protected void shutdown() throws Exception {
		jep.close();
	}
	
	private static Jep jep;

	private Jep initJEP() throws JepException {
		if (jep != null) return jep;
		String pathToScript = System.getProperty("user.dir") 
				+ "/data-collection/lstmTestModelJepVersion.py";
		jep = new Jep(new JepConfig().addSharedModules("tensorflow"));
		jep.runScript(pathToScript);
		return jep;
	}

	@Override
	protected void consume(JEPCall msg, Actor from) throws Exception {
		jep = initJEP();
		
		Object answer = jep.getValue(msg.call);
		msg.output = answer;
	}
	
	
}

class JEPCall {
	
	@Override
	public String toString() {
		return "JEPCall [call=" + call + ", output=" + output + "]";
	}

	final String call;

	public JEPCall(String string) {
		call = string;
	}

	volatile Object output;
}

