from rest_framework import serializers
from marketdata.models import Symbol   

class SymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symbol
        fields = ("id", "symbol", "company_name")
