package com.goodloop.egbot;

import java.io.File;

import com.winterwell.es.ESPath;
import com.winterwell.es.IESRouter;
import com.winterwell.utils.io.Option;
import com.winterwell.web.app.ISiteConfig;

public class EgbotConfig implements IESRouter, ISiteConfig {

	@Option
	private int port = 8641;

	@Override
	public ESPath getPath(CharSequence dataspace, Class type, String id, Object status) {
		String stype = type.getSimpleName().toLowerCase();
		return new ESPath("egbot."+stype, stype,id);
	}

	@Override
	public int getPort() {
		return port;
	}

	public File srcDataDir = new File("data/build"); 
	
}
