package com.goodloop.egbot.tools;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.goodloop.egbot.EgbotConfig;
import com.goodloop.egbot.data.SEQuestion;
import com.winterwell.data.KStatus;
import com.winterwell.es.ESPath;
import com.winterwell.es.IESRouter;
import com.winterwell.es.client.BulkRequestBuilder;
import com.winterwell.es.client.BulkResponse;
import com.winterwell.es.client.ESConfig;
import com.winterwell.es.client.ESHttpClient;
import com.winterwell.es.client.ESHttpRequest;
import com.winterwell.es.client.ESJC;
import com.winterwell.es.client.IESResponse;
import com.winterwell.gson.FlexiGson;
import com.winterwell.gson.Gson;
import com.winterwell.gson.GsonBuilder;
import com.winterwell.gson.KLoopPolicy;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.utils.Dep;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.io.ConfigFactory;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
import com.winterwell.web.app.AppUtils;

public class Step2_CSVIntoES {

	private static final String LOGTAG = "Step2_CSVIntoES";
	private static ESJC esjc;
	private static Gson gson;

	public static void main(String[] args) throws IOException {
		initES();
		// all files
		File buildDataDir = new File("data/build");
		buildDataDir.mkdirs();
		for(File f : buildDataDir.listFiles()) {
			if ( ! f.getName().endsWith(".json")) continue;
			bulkUploadToES(f);
		}
	}

	private static void initES() {
		ConfigFactory cf = ConfigFactory.get();
		ESConfig esc = cf.getConfig(ESConfig.class);
		esjc = new ESJC(esc);
		esjc.debug = false;
		Dep.set(ESHttpClient.class, esjc);
		EgbotConfig egbotConfig = cf.getConfig(EgbotConfig.class);
		Dep.set(IESRouter.class, egbotConfig);
		
		// gson
		gson = new GsonBuilder().setClassProperty(null).setLoopPolicy(KLoopPolicy.NO_CHECKS)
				.create();
		Dep.set(Gson.class, gson);
		
		// let's (ab)use the GL setup code		
		KStatus[] statuses = new KStatus[] {KStatus.PUBLISHED};
		Class[] dbclasses = new Class[] {SEQuestion.class};
		AppUtils.initESIndices(statuses, dbclasses);
		Map<Class, Map> mappingFromClass = new ArrayMap();
		AppUtils.initESMappings(statuses, dbclasses, mappingFromClass);
	}

	private static void bulkUploadToES(File f) throws IOException {
		BulkRequestBuilder bulk = esjc.prepareBulk();
		bulk.setDebug(false);
		IESRouter router = Dep.get(IESRouter.class);
		// use a streaming reader
		JsonReader reader = new JsonReader(FileUtils.getReader(f));
        reader.beginArray();
        while (reader.hasNext()) {
        	SEQuestion q = gson.fromJson(reader, SEQuestion.class);
            String id = q.getId();
            if (id==null) {
            	Log.e(LOGTAG, "No Id?! "+q);
            	continue;
            }
			ESPath path = router.getPath(SEQuestion.class, id);
            ESHttpRequest indexq = esjc.prepareIndex(path);
			indexq.setBodyMap(q);
			bulk.add(indexq); // but we are building this in memory!
        }
        reader.endArray();
        reader.close();
        // call ES
        BulkResponse resp = bulk.get();
        
        resp.check();
        if (resp.hasErrors()) {
        	System.out.println(resp);
        	String json = resp.getJson();
        	System.out.println(json);
        }
	}
}
