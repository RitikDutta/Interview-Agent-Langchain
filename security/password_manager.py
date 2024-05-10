import bcrypt

class Password_manager():
	# hashing
	def hash_password(self, password):
		hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
		return hashed

	#verify
	def verify_password(self, stored_hash, user_password):
	    return bcrypt.checkpw(user_password.encode(), stored_hash)
