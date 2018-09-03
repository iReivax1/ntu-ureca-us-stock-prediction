# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-12-28 19:38
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dbmgr', '0002_auto_20171015_1112'),
    ]

    operations = [
        migrations.CreateModel(
            name='NewspaperFeed',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_datetime', models.DateTimeField()),
                ('text', models.TextField(max_length=400)),
                ('text2', models.TextField(max_length=400, null=True)),
                ('sentiment', models.DecimalField(decimal_places=12, max_digits=13, null=True)),
                ('source', models.TextField(max_length=400)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]