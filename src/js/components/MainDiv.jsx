import React, { Component } from 'react';
import { assert } from 'sjtest';
import Login from 'you-again';
import md5 from 'md5';
import { modifyHash } from 'wwutils';
// Plumbing
import JSend from '../base/data/JSend';
import DataStore from '../base/plumbing/DataStore';
import Roles from '../base/Roles';
import C from '../C';
// import ServerIO from '../plumbing/ServerIO';
// import ActionMan from '../plumbing/ActionMan';
// Widgets
import MessageBar from '../base/components/MessageBar';
import NavBar from '../base/components/NavBar';
import LoginWidget from '../base/components/LoginWidget';
import Misc from '../base/components/Misc';

import CardAccordion, {Card} from '../base/components/CardAccordion';
import PropControl from '../base/components/PropControl';
import BS3 from '../base/components/BS3';

// Pages
import BasicAccountPage from '../base/components/AccountPageWidgets';
import E404Page from '../base/components/E404Page';

class MainDiv extends Component {

	componentWillMount() {
		// redraw on change
		const updateReact = (mystate) => this.setState({});
		DataStore.addListener(updateReact);

		Login.app = C.app.service;
		// Set up login watcher here, at the highest level		
		Login.change(() => {
			// ?? should we store and check for "Login was attempted" to guard this??
			if (Login.isLoggedIn()) {
				// close the login dialog on success
				LoginWidget.hide();
			} else {
				// poke React via DataStore (e.g. for Login.error)
				DataStore.update({});
			}
			this.setState({});
		});

		// Are we logged in?
		Login.verify();
	}

	componentDidCatch(error, info) {
		// Display fallback UI
		this.setState({error, info, errorPath: DataStore.getValue('location', 'path')});
		console.error(error, info); 
		if (window.onerror) window.onerror("Caught error", null, null, null, error);
	}

	render() {
		return (
			<div className="container avoid-navbar">
				<div className="page MyPage">
					<Card>
						<div className="header">
							<div className="header-text">
								<p className="title"><span>EgBot</span></p>
								<p className="subtitle"><span>an example-based question & answer site</span><br/><span> for all of your math needs</span></p>
							</div>
						</div>
					</Card>		
					<Card>
						<div className="row">
							<div className="col-md-6 question-area-wrapper">
								<div className="input-group question-area">
									<div><b>Try out EgBot!</b></div>
									<QuestionForm />
								</div>
								<AnswerPanel />
							</div>
							<div className="col-md-6 play">
								<img className="play-demo" src="https://i.imgur.com/ZR17qnE.png" />
							</div>
						</div>
					</Card>	
				</div>
			</div>
		);
	} // ./render()
} // ./MainDiv


const qpath = ['widget','qform'];

const apath = () => {
	let qform = DataStore.getValue(qpath) || {};
	let qhash = qform.q? md5(qform.q) : 'blank';
	let _apath = ['widget', 'askResponse', qhash];	
	return _apath;
};

const QuestionForm = () => {
	return (
		<div>
			<PropControl 
				path={qpath} prop='q'
				type="textarea" 
				label="Type in your question"
			/>
			<Misc.SubmitButton path={qpath} url='/ask' responsePath={apath()}>Ask</Misc.SubmitButton>
		</div>
	);
};


const AnswerPanel = () => {
	let askResponse = DataStore.getValue(apath()) || {};
	if ( ! askResponse.answer) {
		return (<div className='well'>...</div>);
	}
	let answer = askResponse.answer; 
	return (<div className='well'>
		{answer.body_markdown || answer}
	</div>);
};

export default MainDiv;
