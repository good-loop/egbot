package egbot;

import java.util.List;

import com.winterwell.bob.BuildTask;
import com.winterwell.web.app.build.KPubType;
import com.winterwell.web.app.build.PublishProjectTask;

public class PublishEgBot extends PublishProjectTask {
	
	public PublishEgBot() throws Exception {
		super("egbot", "egbot");
		typeOfPublish = KPubType.production;
	}
	
	@Override
	public List<BuildTask> getDependencies() {
		List<BuildTask> deps = super.getDependencies();		
		return deps;
	}

}
