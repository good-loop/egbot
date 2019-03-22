var path = require('path');
var webpack = require('webpack');

module.exports = {
  entry: ["@babel/polyfill", './src/js/app.jsx'],
  output: { path: __dirname, filename: './web/build/js/bundle.js' },
  devtool: 'source-map',
  resolve: {
    extensions: ['*', '.js', '.jsx'],
	  symlinks: false
  },
  module: {
		rules: [
			{
				test: /.jsx?$/,
				loader: 'babel-loader',
				exclude: /node_modules/,
				options: {
					presets: [
						['@babel/preset-env', { targets: { ie: "11" }, loose: true }]
					],
					plugins: [
						'@babel/plugin-proposal-class-properties',
						'@babel/plugin-transform-react-jsx',
						'transform-node-env-inline'
					]
				}
			}
		],
	},
};
