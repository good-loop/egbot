import React, { Component } from 'react';
import Misc from '../base/components/Misc';

class MainDiv extends Component {



	componentWillMount() {
	}

	render() {
		return (
			<div>
				<div className="container avoid-navbar">
					<div className="page MyPage">
						<Misc.Card>
							<div className="header">
								<div className="header-text">
									<p className="title">EgBot</p>
									<p className="subtitle">A question & answer site<br/> for your math questions.</p>
								</div>
							</div>
						</Misc.Card>
					</div>
				</div>
			</div>
		);
	} // ./render()
} // ./MainDiv

export default MainDiv;
