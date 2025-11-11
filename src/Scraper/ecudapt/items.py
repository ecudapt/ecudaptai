import scrapy

class ForumThread(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    posts = scrapy.Field()