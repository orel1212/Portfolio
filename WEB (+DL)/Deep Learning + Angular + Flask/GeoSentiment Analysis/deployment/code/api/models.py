from api import db

class UserModel(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key = True)
    access_token = db.Column(db.String(120), nullable = False)
    access_secret = db.Column(db.String(120), nullable = False)


    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def update_to_db(cls, user, access_token, access_secret):
        user.access_token = access_token
        user.access_secret = access_secret
        db.session.commit()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.get(id)

    @classmethod
    def get_tokens_by_id(cls, id):
        user = cls.query.get(id)
        access_token = user.access_token
        access_secret = user.access_secret
        return (access_token,access_secret)

    @classmethod
    def return_all(cls):
        def to_json(x):
            return {
                'id': x.id,
                'access_token': x.access_token,
                'access_secret': x.access_secret
            }
        return {'users': list(map(lambda x: to_json(x), UserModel.query.all()))}
