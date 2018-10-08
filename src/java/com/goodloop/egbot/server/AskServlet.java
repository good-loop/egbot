package com.goodloop.egbot.server;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import jep.Jep;
import jep.JepConfig;
import jep.JepException;

import com.winterwell.es.fail.ESException;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.threads.Actor;
import com.winterwell.utils.web.SimpleJson;
import com.winterwell.web.ajax.JSend;
import com.winterwell.web.ajax.JThing;
import com.winterwell.web.app.IServlet;
import com.winterwell.web.app.WebRequest;

public class AskServlet implements IServlet {

	@Override
	public void process(WebRequest state) throws Exception {
		String q = state.get("q");

		Object answer;
		List relatedQs = findRelatedQuestion(q);
		List relatedAs = findRelatedAnswer(relatedQs);
		Object generatedAnswer = generateAnswer(q);
		
		ArrayMap data = new ArrayMap(
			"relatedQs", relatedQs,
			"relatedAs", relatedAs,
			"generatedAnswer", generatedAnswer
			);
		JSend jsend = new JSend(data);
		jsend.send(state);
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
	
	private synchronized Object generateAnswer(String q) throws Exception {
		JEPCall msg = new JEPCall("generateResults('" + q + "')");
		jepActor.send(msg);
		while(msg.output==null) {
			Utils.sleep(10);
		}
		return msg.output;
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

