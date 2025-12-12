from rest_framework.pagination import PageNumberPagination

class SymbolPagination(PageNumberPagination):
    page_size = 50              # default page size
    page_size_query_param = 'page_size'
    max_page_size = 200         # limit max size to protect server
