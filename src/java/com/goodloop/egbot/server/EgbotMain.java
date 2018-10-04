package com.goodloop.egbot.server;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.web.app.AMain;
import com.winterwell.web.app.AppUtils;
import com.winterwell.web.app.ISiteConfig;
import com.winterwell.web.app.JettyLauncher;
import com.winterwell.web.app.MasterServlet;

public class EgbotMain extends AMain<EgbotConfig> {

	public EgbotMain() {
		super("egbot", EgbotConfig.class);
	}
	
	public static void main(String[] args) {
		EgbotMain egbot = new EgbotMain();
		egbot.doMain(args);
	}
	
	@Override
	protected void init2(EgbotConfig config) {
		super.init2(config);
		init3_gson();
		init3_ES();
	}

	@Override
	protected void addJettyServlets(JettyLauncher jl) {
		super.addJettyServlets(jl);
		MasterServlet ms = jl.addMasterServlet();
		ms.addServlet("/ask", AskServlet.class);
	}

}
