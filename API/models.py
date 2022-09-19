from django.db import models

# Create your models here.

class Post2(models.Model):
    post_id = models.IntegerField(null=False, primary_key=True)
    content = models.TextField(null=True)
    senti = models.IntegerField(null=True)
    
    def __str__(self):
        return self.senti
    
    