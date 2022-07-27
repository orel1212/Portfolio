from api import app, jwt
from flask import jsonify, make_response, request, redirect, abort
from flask_jwt_extended import (create_access_token, create_refresh_token, jwt_required, jwt_refresh_token_required,
                                get_jwt_identity, get_raw_jwt)
from api.geosentiment_handler import GeoSentimentHandler
from api.twitter_handler import TwitterHandler
from api.demo_handler import DemoHandler
from api.exceptions import ForbiddenInput, TweepyException
from api.models import UserModel

geosentiment_handler = GeoSentimentHandler()
demo_path = "../demo"
demo_handler = DemoHandler(demo_path)


@app.route("/api/trends", methods=['GET'])
@jwt_required
def get_trends():
    current_user_id = get_jwt_identity()
    tokens = UserModel.get_tokens_by_id(current_user_id)

    twitter_handler = TwitterHandler()

    twitter_handler.set_access_tokens(tokens[0], tokens[1])
    trends = twitter_handler.get_trending_hashtags()
    return jsonify(trends=trends)


@app.route("/api/predictions/<search_input>", methods=['GET'])
@jwt_required
def get_predictions_by_text(search_input):
    current_user_id = get_jwt_identity()
    tokens = UserModel.get_tokens_by_id(current_user_id)
    # print(tokens)

    twitter_handler = TwitterHandler()

    twitter_handler.set_access_tokens(tokens[0], tokens[1])
    tweets = twitter_handler.get_tweets(search_input, max_tweets=18000)
    if len(tweets) != 0:
        geosentiment_handler.compute_predictions(tweets)
    return jsonify(tweets=[tweet.serialize() for tweet in tweets])


@app.route("/api/demo/trends", methods=['GET'])
def get_demo_trends():
    # demo_handler = DemoHandler(demo_path)
    hashtags = demo_handler.get_hashtags()
    return jsonify(trends=hashtags)


@app.route("/api/demo/predictions/<search_input>", methods=['GET'])
def get_demo_predictions(search_input):
    search_input = search_input.lower()
    # demo_handler = DemoHandler(demo_path)
    # demo_handler.load_all_tweets()
    preprocessed_search_input = demo_handler.preprocess_search_input(search_input)
    tweets = demo_handler.search_by_hashtags(preprocessed_search_input)
    if len(tweets) != 0:
        geosentiment_handler.compute_predictions(tweets)
    return jsonify(tweets=[tweet.serialize() for tweet in tweets])


@app.route("/api/twitter_auth")
def twitter_auth():
    twitter_handler = TwitterHandler()
    url = twitter_handler.get_authorization_url()
    return jsonify(request_url=url, request_token=twitter_handler.get_request_token())


@app.route('/api/twitter_verify_auth', methods=['POST'])
def twitter_verify_auth():
    try:
        json_post = request.get_json()
    except Exception:
        raise ForbiddenInput(11, "JSON Format Invalid")
    if json_post is None:
        raise ForbiddenInput(12, "No JSON given")
    twitter_handler = TwitterHandler()
    request_token = json_post['request_token']
    verifier_code = json_post['verifier_code']
    twitter_handler.verify_account(request_token, verifier_code)
    tokens = twitter_handler.get_access_tokens()
    # print(tokens)
    user_id = twitter_handler.get_user_id()
    user = UserModel.find_by_id(user_id)
    # print(user)

    # generating JWT tokens
    access_token_jwt = create_access_token(identity=user_id)
    refresh_token_jwt = create_refresh_token(identity=user_id)
    response = {
        'message': 'Authentication Success',
        'access_token': access_token_jwt,
        'refresh_token': refresh_token_jwt
    }
    # check if we need to update the user
    if user:
        UserModel.update_to_db(user, tokens[0], tokens[1])

        return jsonify(response)

    new_user = UserModel(
        id=user_id,
        access_token=tokens[0],
        access_secret=tokens[1]
    )

    new_user.save_to_db()
    return jsonify(response)


@app.route("/api/refresh_token")
@jwt_refresh_token_required
def token_refresh():
    current_user_id = get_jwt_identity()
    access_token = create_access_token(identity=current_user_id)
    return jsonify({'access_token': access_token})


@app.route('/api/get_name')
@jwt_required
def request_twitter():
    current_user_id = get_jwt_identity()
    tokens = UserModel.get_tokens_by_id(current_user_id)
    twitter_handler = TwitterHandler()
    twitter_handler.set_access_tokens(tokens[0], tokens[1])

    return jsonify(twitter_handler.api.me().name)


@jwt.expired_token_loader
def custom_expired_token_callback():
    return jsonify({
        'code': 4,
        'message': 'The Token Has Expired'
    }), 401


@jwt.invalid_token_loader
def custom_invalid_token_callback(msg):
    return jsonify({
        'code': 5,
        'message': 'The Token is Invalid'
    }), 401


@jwt.unauthorized_loader
def custom_unauthorized_callback(msg):
    return jsonify({
        'code': 6,
        'message': 'Unauthorized Access'
    }), 401


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'code': 3, 'message': "Path Not Found"}), 404)


@app.errorhandler(ForbiddenInput)
def forbidden_input_error(error):
    return make_response(jsonify({'code': error.code, 'message': error.message}), 403)


@app.errorhandler(TweepyException)
def tweepy_exception(error):
    return make_response(jsonify({'code': error.code, 'message': error.message}), error.status_code)


@app.errorhandler(Exception)
def internal_error(error):
    print(error)
    return make_response(jsonify({'code': 1, 'message': "Internal Server Error"}), 500)
