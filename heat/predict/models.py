from django.db import models

# Create your models here.
class Prediction(models.Model):
	음식명 = models.CharField(max_length = 20) 
	음식사진 = models.ImageField(null=True, upload_to='images/', blank=True)

	def __str__(self):
		return self.real

